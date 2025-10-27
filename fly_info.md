# Fly Trajectory Data Info
### Links
- [MABe22 Dataset Website](https://sites.google.com/view/computational-behavior/our-datasets/mabe2022-dataset)
- [Paper](https://arxiv.org/pdf/2207.10553)
    - Appendix B.2. Fly Datasheet
- [Dataset Download](https://data.caltech.edu/records/8kdn3-95j37)
- [Explanation of Data](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-fruit-fly-groups)
- [Dataset from Kai](https://docs.google.com/document/d/116wZ-RpL5NwFL2FWWN9R9kpMdFe9ZeBShsgJsnsAYR0/)

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
- This is my understanding:
    - 101??
        - There are 50 tasks for the fly dataset..
    - 2443500 = 4500 (frames) x 543 (test sequences).

## Additional Info
`"keypoints" shape = (4500, 11, 24, 2)`
- 4500: # frames
- 11: # flies (animal ID) - is anywhere from 9-11 and data are padded with nans to be all the same size
- 24: body part (24 keypoints)
- 2: x, y coordinate

`"annotations" shape = (3, 4500)`
- Labels for 3 sample evaluation tasks: genotype, stimulation, and aggressive female behavior. See [here](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-fruit-fly-groups#Public%20Tasks:~:text=all%20the%20clips.-,Public%20Tasks,-To%20give%20you) for more info.
