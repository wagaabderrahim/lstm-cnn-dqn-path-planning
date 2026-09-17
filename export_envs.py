import os
import numpy as np
import matplotlib.pyplot as plt
from cnn_dqn import GridWorld
import zipfile

def export_environments(num_envs=100, size=15, obstacle_ratio=0.18):
    os.makedirs('env_images', exist_ok=True)
    grids = []
    
    for i in range(num_envs):
        env = GridWorld(size=size, obstacle_ratio=obstacle_ratio, seed=1000 + i)
        env.reset()
        
        # Save grid structure
        grid_data = np.copy(env.grid)
        grids.append(grid_data)
        
        # Plot and save image
        plt.figure(figsize=(5, 5))
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'blue'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Map values for visualization: 0: free, 1: obstacle, 2: start(agent), 3: goal
        vis_grid = np.zeros_like(env.grid)
        vis_grid[env.grid == 1] = 1 # Obstacles
        vis_grid[env.start] = 2 # Start (but wait, agent is at start currently)
        vis_grid[env.goal] = 3 # Goal
        
        plt.imshow(vis_grid, cmap=cmap, norm=norm, origin='upper')
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Test Environment {i+1}')
        plt.savefig(f'env_images/env_{i+1:03d}.png', bbox_inches='tight')
        plt.close()
        
    # Save all grids to npz
    np.savez_compressed('test_environments.npz', grids=np.array(grids))
    
    # Zip the images
    with zipfile.ZipFile('env_images_compressed.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk('env_images'):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)
                
    print("Export complete. Saved test_environments.npz and env_images_compressed.zip")

if __name__ == '__main__':
    export_environments()
