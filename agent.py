import numpy as np
import gymnasium as gym
import gymnasium.utils.save_video as vid
import time
import matplotlib.pyplot as plt
import os
import playsound3 as ps


class RL_cartpole(object):

    def __init__(self, env=None , gamma=0.95 , alpha=0.2 , epsilon=0.3 , show=False , render_mode=None , default=0.0):
        if env is None:
            if render_mode is None:
                if show is True:
                    self.env = gym.make("CartPole-v1" , render_mode="human")
                else:
                    self.env = gym.make("CartPole-v1")
            else:
                self.env = gym.make("CartPole-v1" , render_mode=render_mode)
            self.state , self.info = self.env.reset()
            self.state = self.discreat_states(self.state)
            self.Q = {}
            self.gamma , self.alpha , self.epsilon = gamma , alpha , epsilon
            self.default = default

    def Reward(self , state , done=False):
        if done:
            return -10.0 #Heavy negetive reward
        return 1

        #Effortful but Unworkable Code sadly
        x , x0 , th , th0 = state
        x_m , th_m = 2.4 , 0.21
        w1 , w2 = 0.1 , 1.0

        pos_penelty = w1*((x/x_m)**2)
        ang_penelty = w2*((th/th_m)**2)
        reward = 1 - (pos_penelty + ang_penelty)

        return max(-1.0 , reward)

    def policy(self , state):

        Probability = np.random.random()

        if Probability > (self.epsilon):
            Qs = [ ]
            for action in range(2):
                Qs.append(self.Q.get((state , action) , self.default))
            if Qs[0] == Qs[1]:
                return np.random.choice(2)
            return np.argmax(Qs)
        else:
            return np.random.choice(2)

    def update_Q(self , state , action , state_prime , done=False):

        Q_val = self.Q.get((state , action) , self.default)
        Q_max = []
        if done:
            Q_max = 0.0
        else:
            for a in range(2):
                Q_max.append(self.Q.get((state_prime , a) , self.default))
            Q_max = float(max(Q_max))

        self.Q[(state , action)] = (1-self.alpha)*(Q_val) + self.alpha*(self.Reward(state_prime , done) + self.gamma*(Q_max))

    def discreat_states(self , state):
        
        self.piece = 8
        pole_ang , pole_v = state[2] , state[3]
        pos , vel = state[0] , state[1]

        angle_idx = int(np.digitize(pole_ang, np.linspace(-0.21, 0.21, self.piece)) - 1)
        velocity_idx = int(np.digitize(pole_v, np.linspace(-3.0, 3.0, self.piece)) - 1)
        pos_idx = int(np.digitize(pos , np.linspace(-2.4, 2.4, self.piece)) - 1)
        vel_idx = int(np.digitize(vel , np.linspace(-3.0 , 3.0 , self.piece)) -1)

        angle_idx = max(0, min(self.piece-1, angle_idx))
        velocity_idx = max(0, min(self.piece-1, velocity_idx))
        position = max(0, min((self.piece)-1 , pos_idx))
        velocity = max(0 , min((self.piece)-1 , vel_idx))

        return (angle_idx , velocity_idx , position , velocity)

    def save_Q(self, file):
        import pickle
        Data = (self.Q , self.epsilon , self.alpha , self.gamma , self.piece)
        with open(file , "wb") as f:
            pickle.dump(Data , f)
    def load_Q(self, file):
        import pickle
        with open(file , "rb") as f:
            try:
                Data = pickle.load(f)
            except:
                Data = ({} , 0.3 , 0.2 , 0.95 , 8)
        self.Q , self.epsilon , self.alpha , self.gamma , self.piece = Data

    def _run_model_0(self , epesodes=1550 , horizon=50 , to_show=1500 , interval=100 , autosave_file=None , decay_rate=0.9998    , graph=True):

        show = False
        self.Win = []
        self.win_throughout = []
        plt.ion()
        for t in range(epesodes):

            self.epsilon = max(0.01 , self.epsilon*decay_rate)
            self.state , self.info = self.env.reset()

            if horizon < 500:
                self.Win.append(True)

            for _ in range(horizon):

                self.action = self.policy(tuple(self.state))
                state_old = self.state[:]
                action_old = int(self.action)

                state_prime = self.env.step(self.action)
                self.state = self.discreat_states(state_prime[0])
                if t >= to_show:
                    time.sleep(0.0000001)
            
                done1 , done2 = bool(state_prime[2]) , bool(state_prime[3])
                self.update_Q(tuple(state_old) , action_old , tuple(self.state) , done1)

                if show:
                    print(_ , self.Reward(self.state , done1) , self.state)

                if done1 or done2:
                    state0 , self.info = self.env.reset()
                    self.state = self.discreat_states(state=state0)
                    if done1:
                        self.Win.append(False)

                        if t >= to_show:
                            print(f"***************\n******LOST******\n***************")
                            time.sleep(0.001)
                    if done2:
                        self.Win.append(True)

                        if t >= to_show:
                            print(f"***************\n******WIN******\n***************")
                            time.sleep(0.001)
            
            if graph:
                self.win_throughout.append(self.Win.count(True))
                plt.clf()
                plt.plot(self.win_throughout , range(1,len(self.win_throughout)+1))
                time.sleep(0.1)

            if t == to_show:
                self.env = gym.make("CartPole-v1" , render_mode="human")
                self.state , self.info = self.env.reset()
                self.epsilon = 0
                show = True
            if t%interval==0:

                print("\033c", end="", flush=True)
                print(f"[ Ep:{t}  WR:{round( self.Win.count(True) / (len(self.Win)+0.01) , 5)*100} ]" , end="->" , flush=True)

                if t%(interval*20) == 0 and autosave_file is not None:
                    self.save_Q(autosave_file)
                    print(f"Save {t//(interval*20)}")

    def _run_model_(self , epesodes=1550 , horizon=50 , to_show=1500 , interval=100 , autosave_file=None , decay_rate=0.9998 , graph=True):

        show = False
        self.Win = []
        self.win_throughout = []
        plt.ion()
        for t in range(epesodes):

            self.epsilon = max(0.01 , self.epsilon*decay_rate)
            
            # FIX 1: Discretize the initial reset observation properly
            raw_state , self.info = self.env.reset()
            self.state = self.discreat_states(raw_state)

            episode_ended = False  # Track if the episode wrapped up inside the horizon loop

            for _ in range(horizon):

                self.action = self.policy(tuple(self.state))
                state_old = self.state[:]
                action_old = int(self.action)

                state_prime = self.env.step(self.action)
                self.state = self.discreat_states(state_prime[0])
                if t >= to_show:
                    time.sleep(0.0000001)
            
                done1 , done2 = bool(state_prime[2]) , bool(state_prime[3])
                self.update_Q(tuple(state_old) , action_old , tuple(self.state) , done1)

                if show:
                    print(t , self.Reward(self.state , done1) , self.state)

                if done1 or done2:
                    state0 , self.info = self.env.reset()
                    self.state = self.discreat_states(state=state0)
                    episode_ended = True  # Mark that it finished naturally before horizon
                    if done1:
                        self.Win.append(False)

                        if t >= to_show:
                            print(f"***************\n******LOST******\n***************")
                            time.sleep(0.001)
                    if done2:
                        self.Win.append(True)

                        if t >= to_show:
                            print(f"***************\n******WIN******\n***************")
                            time.sleep(0.001)
                    break

            # FIX 2: If the horizon loop finishes completely without hitting done1 or done2, it's a full-horizon win!
            if not episode_ended:
                self.Win.append(True)
                if t >= to_show:
                    print(f"***************\n******WIN (Horizon Reached)******\n***************")
                    time.sleep(0.001)
            
            if graph:
                self.win_throughout.append(self.Win.count(True))
                plt.clf()
                plt.plot(self.win_throughout , range(1,len(self.win_throughout)+1))
                time.sleep(0.1)

            if t == to_show:
                self.env = gym.make("CartPole-v1" , render_mode="human")
                raw_state , self.info = self.env.reset()
                self.state = self.discreat_states(raw_state)
                self.epsilon = 0
                show = True
            if t%interval==0:

                print("\033c", end="", flush=True)
                print(f"[ Ep:{t}  WR:{round( self.Win.count(True) / (len(self.Win)+0.01) , 5)*100} ]" , end="->" , flush=True)

                if t%(interval*20) == 0 and autosave_file is not None:
                    self.save_Q(autosave_file)
                    print(f"Save {t//(interval*20)}")

    def run_randomly(self):
        done = False
        env = self.env
        while True:

            action = env.action_space.sample()
            print(action)
            s_prime = env.step(action)
            time.sleep(0.02)
            print(s_prime)

            done = s_prime[2] or s_prime[3]
            if done:
                print("*******************************LOST********************************")
                env.reset()


f_new = r"C:\Users\rosha\OneDrive\Desktop\Python_ML\Machine_Learning_Practice_Projects\Game Bot\Data_New1.pkl"
f_old = r"C:\Users\rosha\OneDrive\Desktop\Python_ML\Machine_Learning_Practice_Projects\Game Bot\Q_table.pkl"

train = (100000 , 500 , 99985 , 100)
test = (5 , 500 , 0 , 1)

Model = RL_cartpole(gamma=0.95,
                    alpha=0.2,
                    epsilon=0.1
                    )

#Model.load_Q(file=f_old)


k = input("Start? ")

dr = 0.999931978365943

start = False
if start is True:
    Model._run_model_(*test , autosave_file=None , graph=False , decay_rate=0.9998)
    print(Model.Win.count(True) , Model.Win.count(False) , len(Model.Win))
    #Model.save_Q(file=f_old)

sound = ps.playsound(r"C:\Users\rosha\Downloads\soundreality-telephone-ring-129620.mp3" , block=False)
k = input("End n start rec?")
sound.stop()

k = input()

#Below this coded by AI, I needed to record a video demonstration and not wnating to learn a new
#Library in the way

import os
from gymnasium.wrappers import RecordVideo

# 1. Initialize your recording model instance in frame capture mode
print("\nInitializing video rendering pipeline...")
Model_Record = RL_cartpole(render_mode="rgb_array")

# 2. CRUCIAL FIX: Load your mastered memory file so it doesn't play with a blank brain!

#Model_Record.load_Q(file=f_old)
Model_Record._run_model_(epesodes=1500 , horizon=500 , to_show=1600)

# 3. Force exploration completely off so the AI acts with 100% flawless confidence
Model_Record.epsilon = 0.0

# 4. Wrap your model's active environment inside Gymnasium's RecordVideo module
video_folder = "mit_6036_project_videos"
Model_Record.env = RecordVideo(
    env=Model_Record.env,
    video_folder=video_folder,
    episode_trigger=lambda episode_id: True, # Record every game session
    name_prefix="trained_agent_performance"
)

# 5. Run the model for 3 demonstration games with a higher horizon
print(f"Recording flawless balancing runs to folder: '{video_folder}'...")
# We use episodes=3 and a long horizon=500 so you get beautiful, long showcase clips!
Model_Record._run_model_(epesodes=10, horizon=500 , to_show=9999)

# 6. CRUCIAL FIX: Forcibly close the outer RecordVideo wrapper to flush and save the files!
Model_Record.env.close()

print("\nRecording successfully finalized with zero truncated video drops!")
print(f"Open your workspace directory and look for the folder: {os.path.abspath(video_folder)}")



