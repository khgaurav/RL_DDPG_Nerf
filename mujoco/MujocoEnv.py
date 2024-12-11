import mujoco
import numpy as np
import torch
import os
import gym
from gym import spaces

class FrankaFR3MujocoEnv:
    def __init__(self, model_xml="fr3.xml", image_size=(64, 64)):
        # FR3 has 7 DOF with torque sensors at each joint
        self.n_joints = 7
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data = mujoco.MjData(self.model)
        
        # Setup rendering
        mujoco.glfw.glfw.init()
        window = mujoco.glfw.glfw.create_window(640, 480, "Mujoco", None, None)
        mujoco.glfw.glfw.make_context_current(window)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        self.camera = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.viewport = mujoco.MjrRect(0, 0, *image_size)
        
        # FR3 specific parameters
        self.max_torques = np.array([87, 87, 87, 87, 12, 12, 12])
        self.position_limits = np.array([
            2.7437,   # joint1
            1.7837,   # joint2
            2.9007,   # joint3
            3.0421,   # joint4
            2.8065,   # joint5
            4.5169,   # joint6
            3.0159    # joint7
        ])

    def get_state(self):
        return {
            'joint_pos': self.data.qpos[:self.n_joints].copy(),
            'joint_vel': self.data.qvel[:self.n_joints].copy(),
            'ee_pos': self.data.site('attachment_site').xpos.copy()
        }

    def get_images(self, num_cameras=3):
        images = []
        # Set up multiple virtual cameras around the robot
        camera_positions = [
            (1.5, 0.0, 1.5),  # Front view
            (0.0, 1.5, 1.5),  # Side view
            (-1.0, -1.0, 1.5)  # Diagonal view
        ]
        
        for pos in camera_positions[:num_cameras]:
            self.camera.lookat[:] = [0, 0, 0.5]
            self.camera.distance = 2.0
            self.camera.azimuth = np.arctan2(pos[1], pos[0]) * 180 / np.pi
            self.camera.elevation = -30
            
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.camera,
                                 mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(self.viewport, self.scene, self.context)
            
            image = np.empty((self.viewport.height, self.viewport.width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(image, None, self.viewport, self.context)
            images.append(np.flipud(image))
            
        return np.array(images)

class FR3Encoder(torch.nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        
        # Image processing branch (keep as is)
        self.image_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten()
        )
        
        # State processing branch - match input/output dimensions
        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(14, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 14)  # Output same dimension as input
        )
        
        # Fusion layer
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(128 + 14, latent_dim),
            torch.nn.ReLU()
        )

    def forward(self, images, joint_states):
        # Process multiple camera views
        image_features = []
        for i in range(images.shape[1]):
            img_feat = self.image_encoder(images[:, i])
            image_features.append(img_feat)
        
        # Average features from multiple views
        image_features = torch.stack(image_features, dim=1).mean(dim=1)
        
        # Process joint states - now matches input dimension
        state_features = self.state_encoder(joint_states)
        
        # Combine features
        combined = torch.cat([image_features, state_features], dim=1)
        return self.fusion(combined)


class FR3RLEnv(gym.Env):
    def __init__(self, model_path, encoder):
        super().__init__()
        self.fr3_env = FrankaFR3MujocoEnv(model_path)
        self.encoder = encoder
        
        # FR3 specific action space: 7 joint torques
        self.action_space = spaces.Box(
            low=-1 * self.fr3_env.max_torques,
            high=self.fr3_env.max_torques,
            shape=(7,),
            dtype=np.float32
        )
        
        # Encoded state space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(64,),
            dtype=np.float32
        )
        self.max_steps = 100
        self.current_step = 0

    def step(self, action):
        # Apply torque commands to FR3
        self.fr3_env.data.ctrl[:] = action
        mujoco.mj_step(self.fr3_env.model, self.fr3_env.data)
        
        # Get robot state and images
        state = self.fr3_env.get_state()
        images = self.fr3_env.get_images()
        
        # Encode state
        with torch.no_grad():
            joint_states = torch.FloatTensor(np.concatenate([
                state['joint_pos'],
                state['joint_vel']
            ])).unsqueeze(0)
            images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
            encoded_state = self.encoder(images_tensor, joint_states).numpy()[0]
        
        # Simple reward example based on end-effector position
        reward = -np.linalg.norm(state['ee_pos'] - np.array([0.5, 3, 0.5]))
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return encoded_state, reward, done, {}

    def reset(self):
        mujoco.mj_resetData(self.fr3_env.model, self.fr3_env.data)
        state = self.fr3_env.get_state()
        images = self.fr3_env.get_images()
        
        with torch.no_grad():
            joint_states = torch.FloatTensor(np.concatenate([
                state['joint_pos'],
                state['joint_vel']
            ])).unsqueeze(0)
            images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
            encoded_state = self.encoder(images_tensor, joint_states).numpy()[0]
        
        self.current_step = 0
        return encoded_state
import torch.utils.data as data
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def collect_demonstration_data(env, num_episodes=10, steps_per_episode=50):
    images_data = []
    states_data = []
    
    for episode in tqdm(range(num_episodes), desc="Collecting demonstrations"):
        env.fr3_env.data.qpos[:] = np.random.uniform(
            -env.fr3_env.position_limits,
            env.fr3_env.position_limits
        )
        mujoco.mj_forward(env.fr3_env.model, env.fr3_env.data)
        
        for _ in range(steps_per_episode):
            state = env.fr3_env.get_state()
            images = env.fr3_env.get_images()
            
            images_data.append(images)
            states_data.append(np.concatenate([state['joint_pos'], state['joint_vel']]))
            
            action = env.action_space.sample()
            env.step(action)
    
    return np.array(images_data), np.array(states_data)

def train_encoder(encoder, images, states, num_epochs=50, batch_size=32):
    dataset = data.TensorDataset(
        torch.FloatTensor(images).permute(0, 1, 4, 2, 3) / 255.0,
        torch.FloatTensor(states)
    )
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    for epoch in tqdm(range(num_epochs), desc="Training encoder"):
        total_loss = 0
        for batch_images, batch_states in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            encoded = encoder(batch_images, batch_states)
            
            # Simple reconstruction loss - you might want to modify this
            decoded_states = encoder.state_encoder(batch_states)
            loss = criterion(decoded_states, batch_states)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(dataloader):.6f}")

def visualize_episode(images, episode_rewards, episode_num):
    plt.figure(figsize=(15, 5))
    
    # Plot sample images from different cameras
    for i in range(3):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i])
        plt.title(f'Camera {i+1}')
        plt.axis('off')
    
    # Plot rewards
    plt.subplot(1, 4, 4)
    plt.plot(episode_rewards)
    plt.title(f'Episode {episode_num} Rewards')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.show()

def main():
    os.environ['MUJOCO_GL'] = 'egl'
    # Setup environment and encoder
    model_path = "fr3.xml"  # Make sure this points to your FR3 XML file
    encoder = FR3Encoder(latent_dim=64)
    env = FR3RLEnv(model_path, encoder)
    
    print("Collecting demonstration data...")
    images, states = collect_demonstration_data(env)
    # Display a few images
    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[0, i])
        plt.title(f'Camera {i+1}')
        plt.axis('off')
    plt.show()
    print("Training encoder...")
    train_encoder(encoder, images, states)
    
    # Run random policy episodes
    num_episodes = 5
    steps_per_episode = 100
    
    for episode in range(num_episodes):
        print(f"\nRunning episode {episode + 1}")
        obs = env.reset()
        episode_rewards = []
        episode_images = None
        
        for step in range(steps_per_episode):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            
            # Store first frame for visualization
            if step == 0:
                episode_images = env.fr3_env.get_images()
            
            if done:
                break
            
            # Optional: add small delay to visualize movement
            time.sleep(0.01)
        
        # Visualize episode
        visualize_episode(episode_images, episode_rewards, episode + 1)
        
        print(f"Episode {episode + 1} completed with average reward: {np.mean(episode_rewards):.2f}")

if __name__ == "__main__":
    main()
