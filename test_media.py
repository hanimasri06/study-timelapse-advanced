import asyncio
from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager

async def test():
    print("Requesting sessions...")
    try:
        sessions = await GlobalSystemMediaTransportControlsSessionManager.request_async()
        print("Got sessions:", sessions)
        current_session = sessions.get_current_session()
        print("Current session:", current_session)
        if current_session:
            info = await current_session.try_get_media_properties_async()
            print("Title:", info.title)
            print("Artist:", info.artist)
    except Exception as e:
        print("Exception:", e)

asyncio.run(test())
