#!/bin/sh

case "$1" in
    start)
        pactl load-module module-native-protocol-tcp \
            port=4656 listen=0.0.0.0
        
        pgrep -f pipewire-pulse | while read -r pid; do
            chrt -f -p 70 "$pid" 2>/dev/null
        done
        ;;
    stop)
        pactl unload-module module-native-protocol-tcp
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac
