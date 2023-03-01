# make_dataset
Code for the make_dataset package

This is verion 0.0 for my project

Example for documentation + doc strings

```
"""
Base class for video clips.
    See ``VideoFileClip``, ``ImageClip`` etc. for more user-friendly classes.
    
    Parameters
    ----------
    is_mask
      `True` if the clip is going to be used as a mask.
      
    Attributes
    ----------
    size
      The size of the clip, (width,height), in pixels.
    w, h
      The width and height of the clip, in pixels.
    is_mask
      Boolean set to `True` if the clip is a mask.
    make_frame
      A function ``t-> frame at time t`` where ``frame`` is a
      w*h*3 RGB array.
    mask (default None)
      VideoClip mask attached to this clip. If mask is ``None``,
                The video clip is fully opaque.
    audio (default None)
      An AudioClip instance containing the audio of the video clip.
    pos
      A function ``t->(x,y)`` where ``x,y`` is the position
      of the clip when it is composed with other clips.
      See ``VideoClip.set_pos`` for more details
    relative_pos
      See variable ``pos``.
    layer
      Indicates which clip is rendered on top when two clips overlap in
      a CompositeVideoClip. The highest number is rendered on top.
      Default is 0.
    """
    ```
