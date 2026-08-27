import os
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    langchain,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
    bey,
)

from livekit.plugins.turn_detector.multilingual import MultilingualModel

from graph import create_workflow

load_dotenv(".env.local")


class InterviewAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a professional interviewer conducting a job interview. "
                "The LangGraph workflow will drive the conversation flow. "
                "Simply speak the questions and responses as they come from the graph. "
                "Be conversational, professional, and helpful throughout the interview process."
            )
        )


async def entrypoint(ctx: agents.JobContext):
    print("\n==============================")
    print("JOB STARTED")
    print("Room:", ctx.room.name)

    required_keys = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "BEY_API_KEY",
        "DEEPGRAM_API_KEY",
        "OPENAI_API_KEY",
        "CARTESIA_API_KEY",
    ]

    for key in required_keys:
        print(f"{key}: {'OK' if os.getenv(key) else 'MISSING'}")

    print("==============================\n")

    print("1. Creating LangGraph workflow...")
    interview_workflow = create_workflow()
    print("✅ LangGraph created")

    lg_llm = langchain.LLMAdapter(
        graph=interview_workflow
    )

    print("2. Creating AgentSession...")

    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
        ),
        llm=lg_llm,
        tts=cartesia.TTS(
            model="sonic-2",
            voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    print("✅ AgentSession created")

    print("3. Creating Beyond Presence AvatarSession...")

    avatar = bey.AvatarSession(
        avatar_id="694c83e2-8895-4a98-bd16-56332ca3f449",
        api_key=os.getenv("BEY_API_KEY"),
    )

    print("✅ AvatarSession object created")
    print("Avatar identity:", avatar.avatar_identity)

    print("4. Starting Beyond Presence avatar...")

    try:
        await avatar.start(
            session,
            room=ctx.room,

            # Explicitly pass these while debugging
            livekit_url=os.getenv("LIVEKIT_URL"),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY"),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )

        print("✅ avatar.start() completed")

    except Exception as e:
        print("\n❌ avatar.start() FAILED")
        print("Exception type:", type(e).__name__)
        print("Exception:", repr(e))
        raise

    print("5. Waiting for avatar to join and publish video...")

    try:
        await avatar.wait_for_join(timeout=30)
        print("✅ Avatar joined and published video")
    except Exception as e:
        print("\n❌ Avatar did not publish video")
        print("Exception type:", type(e).__name__)
        print("Exception:", repr(e))
        raise

    print("\nRemote participants:")

    for identity, participant in ctx.room.remote_participants.items():
        print("Participant:", identity)

        for sid, publication in participant.track_publications.items():
            print(
                "  Track SID:",
                sid,
                "Kind:",
                publication.kind,
                "Name:",
                publication.name,
            )

    print("6. Starting LiveKit AgentSession...")

    await session.start(
        room=ctx.room,
        agent=InterviewAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    print("✅ AgentSession started")
    print("Starting interview workflow...")


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )