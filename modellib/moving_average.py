# Moving average model
def ma_model(data: pd.DataFrame, window_size: int=4*3, shift_size: int=96):
    moving_avg = data.rolling(window=window_size).mean()
    shifted_moving_avg = moving_avg.shift(shift_size)
    return(shifted_moving_avg)