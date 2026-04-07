import gymnasium as gym
import numpy as np

class AgriGuardEnv(gym.Env):
    def __init__(self):
        # Observation: [Janwar ki doori (0-50m), Janwar ka size (1-10)]
        self.observation_space = np.array([50.0, 10.0])
        self.action_space = gym.spaces.Discrete(4) # 0:Stay, 1:Light, 2:Siren, 3:Shock
        self.state = None

    def reset(self):
        # Naya janwar random doori aur size ke saath
        self.state = np.array([np.random.uniform(10, 50), np.random.uniform(1, 10)])
        return self.state

    def step(self, action):
        distance, size = self.state
        reward = +10
        done = False

        # LOGIC:
        # Action 1 (Light): Chote pakshiyon ke liye
        # Action 2 (Siren): Kutte ya Hiran ke liye
        # Action 3 (Shock): Neelgai jaise bade janwaro ke liye
        if action == 0: # Kuch nahi kiya
            reward = -10 # Penalty thodi badhao
        elif action == 1: # Light
            if size < 3:
                reward = 10
                done = True # Janwar bhag gaya!
            else:
                reward = -5 # Bade janwar par asar nahi hua
        elif action == 2: # Siren
            if 3 <= size < 7:
                reward = 15
                done = True # Janwar bhag gaya!
            else:
                reward = -10
        elif action == 3: # Mild Shock
            if size >= 7:
                reward = 20
                done = True # Janwar bhag gaya!
            else:
                reward = -30

        # ASLI FIX: Distance logic aur Khet barbaad hone ki condition
        if not done:
            self.state[0] -= 5 # Janwar paas aa raha hai
            
            # Agar janwar khet mein ghus gaya (Distance 0 ya usse kam)
            if self.state[0] <= 0:
                reward = -100 # Bada nuksaan!
                done = True # Episode khatam kyunki khet barbaad ho gaya
            if self.state[0] <= 0: # Janwar khet mein ghus gaya!
                reward = -100
                done = True

        return self.state, reward, done, {}

# Graders for Task 1 (Easy), Task 2 (Medium), Task 3 (Hard)
def grade_task(score):
    return min(1.0, max(0.0, score / 100.0))

import random

if __name__ == "__main__":
    env = AgriGuardEnv()
    
    # AI ka Dimaag (Q-Table): [States, Actions]
    # Humne 100 states (doori) aur 4 actions rakhe hain
    q_table = np.zeros([101, 4]) 
    
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 1.0  # Shuruat mein AI random khelega
    
    print("--- Agri-Guard AI 'Dimaag' Training Shuru ---")
    
    for episode in range(6000): # Ab hum 1000 baar train karenge!
        state = env.reset()
        dist = int(state[0]) # Doori ko index banaya
        done = False
        
        while not done:
            # Epsilon-Greedy: Seekhne ka tarika
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore (Naya seekho)
            else:
                action = np.argmax(q_table[dist]) # Exploit (Jo pata hai wo karo)

            next_state, reward, done, _ = env.step(action)
            next_dist = int(next_state[0])

            # --- ASLI LEARNING (Q-Update) ---
            old_value = q_table[dist, action]
            next_max = np.max(q_table[next_dist])
            
            # Formula: Naya Dimaag = Purana + Seekha hua
            q_table[dist, action] = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
            
            dist = next_dist
        
        # Dhire dhire random buttons dabana kam karo
        epsilon = max(0.01, epsilon * 0.995)

        if episode % 100 == 0:
            print(f"Episode {episode}: AI seekh raha hai...")

    print("\n--- Training Khatam! AI ab Expert ban gaya hai! ---")
    # --- TEST CODE START ---
    print("\n--- ASLI TEST: Expert AI Khet Ki Rakhwali Kar Raha Hai ---")
    state = env.reset()
    dist = int(state[0])
    done = False
    
    while not done:
        action = np.argmax(q_table[dist]) 
        state, reward, done, _ = env.step(action)
        dist = int(state[0])
        
        print(f"Janwar ki Doori: {dist}m | AI ne liya Action: {action} | Reward: {reward}")
        
        
           # --- FINAL SUCCESS TEST CODE ---
    print("\n--- ASLI TEST: Expert AI Check ---")
    state = env.reset()
    dist = int(state[0])
    done = False
    
    while not done:
        action = np.argmax(q_table[dist]) 
        state, reward, done, _ = env.step(action)
        
        print(f"Doori: {dist}m | Action: {action} | Reward: {reward}")
        dist = int(state[0]) # Doori update karo
        
        if done:
            # Agar reward positive hai (+10, +15, +20), matlab AI ne sahi action liya!
            if reward > 0:
                print("\nSUCCESS: AI ne sahi action liya aur khet bacha liya!")
            else:
                print("\nFAILURE: Janwar khet mein ghus gaya ya galat action liya.")
            break
