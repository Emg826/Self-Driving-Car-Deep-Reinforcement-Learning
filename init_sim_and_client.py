import time
import subprocess
import signal

abs_file_path_to_city_env_bat_file = 'C:\\EricGabriel\\Unreal-AirSim-Environments\\CityEnviron\\CityEnviron\\Binaries\\Win64\\CityEnviron.exe  -ResX=640 -ResY=480 -windowed'
abs_file_path_to_python_client_file = 'C:\\EricGabriel\\Reinforcement-Learning-Project\\kerasrl_dqn_collision_avoidance.py'

# outer loop handles restarts while the
# inner loop handles check ups on the sim
secs_to_wait_for_sim_to_load_up = 30
secs_between_checks = 4
init_counter = 0
running_sims = []
running_clients = []


while True:
  # try to prevent > 1 sim or > 1 client from running @ a time
  # garbage collection to prevent 10 sims from running @ once
  if len(running_sims) != 0:
    for idx in range(0, len(running_sims)):
      running_sims[idx].terminate()
      running_sims[idx].wait()
  running_sims = []
      
  if len(running_clients) != 0:
    for idx in range(0, len(running_clients)):
        running_clients[idx].terminate()
        running_clients[idx].wait()
  running_clients = []

  
  init_counter += 1
  # open the simulation
  print('Opening the airsim city environemnt')
  running_sims.append(subprocess.Popen(abs_file_path_to_city_env_bat_file, shell=True))

  print('Waiting {} seconds for the environment to get set up'.format(secs_to_wait_for_sim_to_load_up))
  time.sleep(secs_to_wait_for_sim_to_load_up)

  print('Starting the client')
  running_clients.append(subprocess.Popen(abs_file_path_to_python_client_file, shell=True))

  want_to_break = False
  while True:
    time.sleep(secs_between_checks)
    
    # run until see that sim has quit
    if running_sims[0].poll() != None:
      running_clients[0].terminate()
      running_clients[0].wait()
      running_sims[0].terminate()
      running_sims[0].wait()

      want_to_break = True

    if running_clients[0].poll() != None:
      running_sims[0].terminate()
      running_sims[0].wait()
      running_clients[0].terminate()
      running_clients[0].wait()

      want_to_break = True

    if want_to_break:
      break
    
  print('just finished initialization {}'.format(init_counter))

