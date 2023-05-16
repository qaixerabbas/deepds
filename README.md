# Deep Dataset
Code for the make_dataset package
need to rename this package as deepds => deepdatasets

This is verion 0.0 for my project

Example for documentation + doc strings

```
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
      ```

### Another example of docstring

```
    Adds basic date-based features based to the data frame.
    --------------------
    Arguments:
    - df (pandas DF): dataset
    - date_var (str): name of the date feature
    - drop (bool): whether to drop the original date feature
    - time (bool): whether to include time-based features
    --------------------
    Returns:
    - pandas DF with new features
    
    --------------------
    Examples:
    
    # create data frame
    data = {'age': [27, np.nan, 30], 
            'height': [170, 168, 173], 
            'gender': ['female', 'male', np.nan],
            'date_of_birth': [np.datetime64('1993-02-10'), np.nan, np.datetime64('1990-04-08')]}
    df = pd.DataFrame(data)
    # add date features
    from dptools import add_date_features
    df_new = add_date_features(df, date_vars = 'date_of_birth')
```    
