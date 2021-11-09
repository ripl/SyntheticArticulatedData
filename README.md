# Synthetic Articulated Data
Procedurally generated articulated objects specified in Universal Robot Description Format (URDF), and rendered using MuJoCo or PyBullet.

# Mujoco
Setup:  
```pip install -r requirements.txt```

Example generation:  
```python generate_data.py --n 10 --dir ./test --obj microwave --masked --debug```

# PyBullet
Setup:  
```pip install -r requirements-bullet.txt```

Example generation:  
```python generate_data.py --n 10 --dir ~/data/al/articulated_motion --obj microwave --masked --debug --pybullet --mode 1```
```python generate_data.py --n 10 --dir ~/data/al/camera_motion --obj microwave --masked --debug --pybullet --mode 2```
