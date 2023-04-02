# from vidgear.gears import CamGear
# import cv2

# stream = CamGear(
#     source="https://www.youtube.com/watch?v=oQyKL_jBz0Q&ab_channel=MLTNA7X",
#     stream_mode=True,
#     time_delay=1,
#     logging=True,
# ).start()

# while True:

#     frame = stream.read()
#     if frame is None:
#         break

#     cv2.imshow("Output Frame", frame)  # optional if u want to show the frames

# import required libraries
from vidgear.gears import CamGear
import cv2

# Add YouTube Video URL as input source (for e.g https://youtu.be/bvetuLwJIkA)
# and enable Stream Mode (`stream_mode = True`)
stream = CamGear(
    source="https://www.youtube.com/watch?v=WX7bNiLofIA",
    stream_mode=True,
    logging=True,
).start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
