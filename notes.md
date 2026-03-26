tv feature

Good question. Let me think through this.

The core problem: The Jetson TV currently always shows the camera effects output. You want the phone to control what's on the TV — camera effects, a YouTube video, or eventually user-submitted content (uploaded videos, face swaps).

My recommendation: a "TV Source" switcher

Instead of tying it to phone UI tabs, add a concept of TV source — what the Jetson is actually outputting to the display. The phone controls it, but they're independent concerns.

TV sources:

Camera FX — what it does now (default)
YouTube — plays a video fullscreen on the TV via mpv
Media (future) — plays an uploaded video or shows a face swap result
How it works on the phone:

A small persistent bar at the top (below the status bar) shows the current TV source with quick-switch buttons: [Camera FX] [YouTube]
Tapping YouTube doesn't change the phone tab — it just tells the Jetson to switch its output
You can still browse any phone tab independently (search YouTube while camera FX is on TV, etc.)
How it works on the Jetson:

The preview window (mpv or OpenCV) switches based on the active TV source
Camera FX: shows the normal effects pipeline output
YouTube: mpv takes over fullscreen (already built)
Media: mpv plays a local file
Why this is better than tab-syncing:

Phone tabs and TV output are separate concerns — you want to search YouTube results while camera FX is still live on TV
Multiple phones can be connected; you don't want one person's tab browsing to hijack the TV
Future sources (uploaded video, face swap output) slot in cleanly without needing new tabs
For uploaded videos / face swaps later:

Upload goes to a "media queue" on the Jetson
Switching TV source to "Media" plays the latest item
Same switcher bar, just add a third button when content is available
Implementation would be:

New engine state: tv_source ("camera" | "youtube" | "media")
New API endpoint: POST /api/tv-source
Phone UI: source switcher bar (always visible, ~40px)
Jetson side: when source is "youtube", hide the OpenCV preview window and let mpv run fullscreen; when "camera", kill mpv and show preview
Want me to build this?