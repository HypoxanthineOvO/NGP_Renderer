import os


scenes = ["chair", "drums","ficus","hotdog","lego","materials","mic","ship"]

for s in scenes:
    os.system(f"python main.py --scene {s}")