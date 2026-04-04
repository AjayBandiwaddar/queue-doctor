# Copyright (c) Ajay Bandiwaddar — OpenEnv Hackathon Round 1
"""
FastAPI server for Queue Doctor.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .queue_environment import QueueDoctorEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.queue_environment import QueueDoctorEnvironment

# Pass class (not instance) for WebSocket session isolation.
# Each connected client gets its own QueueDoctorEnvironment instance.
app = create_app(
    QueueDoctorEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="queue_doctor",
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()