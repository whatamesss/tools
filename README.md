# tools
this repo contains helpers i've found, written, or coached a robot into writing


- force-playbin-flags-23.patch - this file removes network buffering/preroll in Qt6's obnoxious gstreamer pipeline. on gentoo, you can drop this file directly into /etc/portage/patches/media-libs/gst-plugins-bad/ and rebuild gst-plugins-bad. reasoning: i use cantata to control a remote mpd daemon, and i listen to the audio stream via http. cantata uses qtmultimedia for playback which has an ffmpeg backend that is terrible at network audio. switching to the gstreamer backend plays beautifully, but a 3-5 second buffer was being forced with no control over it, thanks Qt! it was a long journey to figure out this simple fix.. --flags=23 is key here.

- 70-rtrpio.start - this is a start script in my /etc/local.d/ that re-prioritizes certain processes/irqs in my realtime scheduler. use it as an example if it helps you..
