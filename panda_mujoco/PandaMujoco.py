import mujoco
import numpy as np
from pathlib import Path
import json
import cv2
import gym
from gym import spaces
import torch
import os

class FrankaMujocoEnv:
    def __init__(self, num_cameras=8):
        # Load the Franka XML model
        self.model = mujoco.MjModel.from_xml_path("panda_nohand.xml")
        self.data = mujoco.MjData(self.model)
        # Setup rendering
        mujoco.glfw.glfw.init()
        window = mujoco.glfw.glfw.create_window(640, 480, "Mujoco", None, None)
        mujoco.glfw.glfw.make_context_current(window)
        # Setup virtual cameras in a circle around the robot
        self.cameras = []
        for i in range(num_cameras):
            angle = (2 * np.pi * i) / num_cameras
            cam_pos = np.array([
                1.5 * np.cos(angle),
                1.5 * np.sin(angle),
                1.0
            ])
            self.cameras.append({
                'pos': cam_pos,
                'target': np.array([0, 0, 0.5]),
                'up': np.array([0, 0, 1])
            })
    
    def _get_camera_matrix(self, pos, target, up):
        # Create a view matrix for the camera
        forward = target - pos
        forward /= np.linalg.norm(forward)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        
        view_matrix = np.eye(4)
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = forward
        view_matrix[:3, 3] = pos
        
        return view_matrix
    
    def render_cameras(self, width=640, height=480):
        images = []
        camera_poses = []
        
        # Create scene and camera objects
        scene = mujoco.MjvScene(self.model, maxgeom=10000)
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        viewport = mujoco.MjrRect(0, 0, width, height)
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        
        for camera_config in self.cameras:
            # Create renderer for each camera
            # renderer = mujoco.Renderer(self.model, height, width)
            
            # Forward simulation to ensure state is updated
            mujoco.mj_forward(self.model, self.data)
            
            # Set camera parameters
            cam.lookat[:] = camera_config['target']
            cam.distance = np.linalg.norm(camera_config['pos'] - camera_config['target'])
            cam.azimuth = np.arctan2(camera_config['pos'][1], camera_config['pos'][0]) * 180 / np.pi
            cam.elevation = np.arctan2(camera_config['pos'][2], 
                                    np.sqrt(camera_config['pos'][0]**2 + camera_config['pos'][1]**2)) * 180 / np.pi
            
            # Update scene with correct argument order
            mujoco.mjv_updateScene(
                self.model,              # Model
                self.data,               # Data
                opt,                     # Option
                None,                    # Perturbation (None if not used)
                cam,                     # Camera
                mujoco.mjtCatBit.mjCAT_ALL.value,  # Category mask
                scene                    # Scene
            )
            mujoco.mjr_render(viewport, scene, context)
            
            image = np.empty((viewport.height, viewport.width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(image, None, viewport, context)
            images.append(np.flipud(image))
            camera_poses.append(self._get_camera_matrix(camera_config['pos'], 
                                                    camera_config['target'], 
                                                    camera_config['up']))
        
        return images, camera_poses





def prepare_nerfstudio_data(images, camera_poses, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save images and create transforms.json
    transforms = {
        "frames": [],
        "camera_angle_x": 0.8,  # Adjust based on your camera FOV
        "camera_angle_y": 0.6,
    }
    
    for idx, (image, pose) in enumerate(zip(images, camera_poses)):
        # Save image
        img_path = output_dir / f"image_{idx:03d}.png"
        cv2.imwrite(str(img_path), image)
        
        # Add transform
        transforms["frames"].append({
            "file_path": f"image_{idx:03d}",
            "transform_matrix": pose.tolist(),
            "fl_x": 800,  # Focal length X
            "fl_y": 800,  # Focal length Y
            "cx": 320,    # Principal point X
            "cy": 240,    # Principal point Y
        })
    
    # Save transforms.json
    with open(output_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)

class FrankaNeRFEnv(gym.Env):
    def __init__(self, nerf_model_path):
        super().__init__()
        
        # Load NeRF model
        self.nerf_model = torch.load(nerf_model_path)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(7,), dtype=np.float32)  # 7 joint angles
        
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            'desired_goal': spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'achieved_goal': spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })
        
        self.reset()
    
    def reset(self):
        # Reset joint positions
        self.joint_positions = np.zeros(7)
        
        # Generate random target point
        self.target_position = self._sample_goal()
        
        return self._get_obs()
    
    def step(self, action):
        # Apply action (joint velocities)
        self.joint_positions += action * 0.05  # Small step size
        
        # Get end-effector position using forward kinematics
        achieved_goal = self._get_ee_position()
        
        # Calculate reward
        reward = self._compute_reward(achieved_goal, self.target_position)
        
        # Check if done
        done = np.linalg.norm(achieved_goal - self.target_position) < 0.05
        
        return self._get_obs(), reward, done, {}
    
    def _compute_reward(self, achieved_goal, desired_goal):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return -distance
    
    def _get_obs(self):
        return {
            'observation': self.joint_positions.copy(),
            'desired_goal': self.target_position.copy(),
            'achieved_goal': self._get_ee_position()
        }


if __name__ == "__main__":
    os.environ['MUJOCO_GL'] = 'egl'
    # 1. Render images from Mujoco
    env = FrankaMujocoEnv(num_cameras=8)
    images, camera_poses = env.render_cameras()

    # 2. Prepare data for Nerfstudio
    prepare_nerfstudio_data(images, camera_poses, "nerf_data")

    # 3. Train NeRF model (command line)
    # ns-train nerfacto --data /path/to/nerf_data

    # # 4. Create and use RL environment
    # rl_env = FrankaNeRFEnv("path_to_trained_nerf_model.pth")
    # obs = rl_env.reset()

    # # 5. Run RL training (example using Stable Baselines3)
    # from stable_baselines3 import SAC

    # model = SAC("MultiInputPolicy", rl_env, verbose=1)
    # model.learn(total_timesteps=10000)
