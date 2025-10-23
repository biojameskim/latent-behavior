# Fly Trajectory Data Info
### Links
- [Behavior Data from Caltech](https://data.caltech.edu/records/8kdn3-95j37)
- [Explanation of Data](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-fruit-fly-groups)

### Data schema:
```
{
    "keypoint_vocabulary": a list of names describing the tracked keypoints
    "vocabulary" : A list of public task names
    "sequences" : {
        "<sequence_id> : {
            "keypoints" : a ndarray of shape (4500, 11, 24, 2)
            "annotations" : a ndarray of shape (3, 4500)
        }
    }
}
```


## `fly_group_train.npy`
- train keys: ['keypoint_vocabulary', 'vocabulary', 'sequences']
- train sequences: 426 records
- train first id: 01FJRKCP4GE1W1DFX51C
- train keypoints sample: shape=(4500, 11, 24, 2), dtype=float32
- train annotations sample: shape=(3, 4500), dtype=float32

## `fly_group_test.npy`
- test keys: ['sequences']
- test sequences: 543 records
- test first id: W3WU4RTQM3E7XEHTWZPB
- test keypoints sample: shape=(4500, 11, 24, 2), dtype=float32

## `fly_groups_test_labels.npy`
- test_labels array: shape=(101, 2443500), dtype=float32

## Additional Info
keypoints shape = (4500, 11, 24, 2)
- 4500: # frames
- 11: # flies (animal ID) - is anywhere from 9-11 and data are padded with nans to be all the same size
- 24: body part (24 keypoints)
- 2: x, y coordinate
