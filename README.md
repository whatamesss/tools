# tools
this repo contains helpers i've found, written, or coached a robot into writing


- force-playbin-flags.patch - this source patch removes network buffering/preroll in Qt6's obnoxious gstreamer pipeline. on gentoo, you can drop this file directly into /etc/portage/patches/media-libs/gst-plugins-bad/ and rebuild/emerge gst-plugins-bad(1.24.13). reasoning: i use cantata to control a remote (but local) mpd daemon, and i listen to the audio stream via http. cantata uses qtmultimedia for playback which has an ffmpeg backend that is terrible at network audio. switching to the gstreamer backend plays beautifully, but a 3-5 second buffer was being forced with no control over it, thanks Qt! it was a long journey to figure out this simple fix.. --flags=2 is for audio-only, no buffers.

- 70-rtrpio.start - this is a startup script in my /etc/local.d/ that re-prioritizes certain processes/irqs in my realtime scheduler. use it as an example if it helps you..

- add-stream-button.patch - this is a patch for cantata(3.4.0) to add a nice toggle button for the http stream consumer, this is equivalent to the menu item checkbox but in a more convenient location. of course, this all depends on having 'streaming' enabled in your build.

- photo_search.py - this one's really cool. it uses cuda/compute on nvidia to index photos according to their content, and presents a gui that allows for searches. it requires python 3.13, and some other things...at the time of writing i've got the following related gentoo packages:
 sci-ml/pytorch-2.11.0:0
 sci-ml/transformers-5.3.0:0
 dev-python/numpy-2.3.2:0/2
 dev-python/pillow-12.1.1:0
 dev-python/pyqt6-6.10.2:0
 x11-drivers/nvidia-drivers-580.142:0/580
 dev-util/nvidia-cuda-toolkit-12.9.1-r1:0/12.9.1
currently missing due to dependency issues is 'torchvision' it allows a bit of a speedup, the script will detect and use it automatically. if/when that package gets fixed, i will update this comment. 
