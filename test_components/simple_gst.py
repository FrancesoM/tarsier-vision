import gi
import time
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)


pipeline = 'rtspsrc location="rtsp://admin:fra_GREG2@10.10.1.108:554/cam/realmonitor?channel=1&subtype=0" latency=100 protocols=tcp ! rtpjitterbuffer ! rtph265depay ! h265parse ! splitmuxsink location=/dev/shm/highres_segments/segment_%06d.mkv max-size-time=10000000000'

print(pipeline)
pipeline_obj=Gst.parse_launch(pipeline)
pipeline_obj.set_state(Gst.State.PLAYING)
while(True):
    time.sleep(1)
