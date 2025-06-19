# Run Aloha Sim

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/aloha_sim/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_sim/alohasim_venv
source examples/aloha_sim/alohasim_venv/bin/activate
uv pip sync examples/aloha_sim/requirements.txt
uv pip install -e packages/openpi-client

# Run the simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

Note: If you are seeing EGL errors, you may need to install the following dependencies:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```
Note: If you see erors INFO:OpenGL.acceleratesupport:No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'
```bash
python -m pip install PyOpenGL_accelerate
```
Note: if you see error 'results = tree.map_structure(lambda x: x[self._cur_step, ...], self _last_results)
TypeError: 'float' object is not subscriptable'
```bash
go to '/home/sl/sl/openpi/packages/openpi-client/src/openpi_client/action_chunk_broker.py'. 
then replace 'results = tree.map_structure(lambda x: x[self._cur_step, ...], self _last_results)' (Line 33) with
'
results = {
        "actions": self._last_results["actions"][self._cur_step, ...],
        "policy_timing": self._last_results["policy_timing"],
        "server_timing": self._last_results["server_timing"],
    }
'
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env ALOHA_SIM
```
