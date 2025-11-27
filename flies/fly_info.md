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
This is a **dictionary** with 5 keys:

1. **`frame_number_map`** (dict with 543 entries)
   - Maps each sequence ID to a tuple of (start_frame, end_frame) indices
   - Each sequence has 4500 frames: (0, 4500), (4500, 9000), (9000, 13500), etc.
   - Used to index into the flattened label_array

2. **`label_array`** (numpy array, shape=(101, 2443500), dtype=float32)
   - 101 rows: one per task/class label (matches vocabulary length)
   - 2,443,500 columns: 4500 frames × 543 test sequences (all frames flattened)
   - Contains binary labels: 0 (negative), 1 (positive), or NaN (unlabeled/not applicable)
   - Most entries are NaN since each task only applies to a subset of sequences

3. **`vocabulary`** (list of 101 items)
   - Task/class names: ['control', 'BDP_sexseparated', 'Control_RGB', '71G01', 'male71G01_femaleBDP', ...]
   - Corresponds to the 101 rows in label_array

4. **`task_type`** (list of 120 items)
   - Indicates task type for each task (all appear to be 'Discrete' for binary classification)
   - Note: 120 items while vocabulary has 101 (some tasks may be hidden/additional)

5. **`keypoint_vocabulary`** (list of 24 tuples)
   - Same as in training data - defines the 24 tracked keypoints

## Additional Info
`"keypoints" shape = (4500, 11, 24, 2)`
- 4500: # frames
- 11: # flies (animal ID) - is anywhere from 9-11 and data are padded with nans to be all the same size
- 24: body part (24 keypoints - description [here](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-fruit-fly-groups#:~:text=Body%20parts%20are,leg%20tip.))
- 2: x, y coordinate

`"annotations" shape = (3, 4500)`
- Labels for 3 sample evaluation tasks: genotype, stimulation, and aggressive female behavior. See [here](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-fruit-fly-groups#Public%20Tasks:~:text=all%20the%20clips.-,Public%20Tasks,-To%20give%20you) for more info.

train: `keypoint_vocabulary` 
```
[('wing_left_x', 'wing_left_y'), ('wing_right_x', 'wing_right_y'), ('antennae_x_mm', 'antennae_y_mm'), ('right_eye_x_mm', 'right_eye_y_mm'), ('left_eye_x_mm', 'left_eye_y_mm'), ('left_shoulder_x_mm', 'left_shoulder_y_mm'), ('right_shoulder_x_mm', 'right_shoulder_y_mm'), ('end_notum_x_mm', 'end_notum_y_mm'), ('end_abdomen_x_mm', 'end_abdomen_y_mm'), ('middle_left_b_x_mm', 'middle_left_b_y_mm'), ('middle_left_e_x_mm', 'middle_left_e_y_mm'), ('middle_right_b_x_mm', 'middle_right_b_y_mm'), ('middle_right_e_x_mm', 'middle_right_e_y_mm'), ('tip_front_right_x_mm', 'tip_front_right_y_mm'), ('tip_middle_right_x_mm', 'tip_middle_right_y_mm'), ('tip_back_right_x_mm', 'tip_back_right_y_mm'), ('tip_back_left_x_mm', 'tip_back_left_y_mm'), ('tip_middle_left_x_mm', 'tip_middle_left_y_mm'), ('tip_front_left_x_mm', 'tip_front_left_y_mm'), ('x_mm', 'y_mm'), ('cos_ori', 'sin_ori'), ('maj_ax_mm', 'min_ax_mm'), ('body_area_mm2', 'fg_area_mm2'), ('img_contrast', 'min_fg_dist')]
```

train: `vocabulary` 
```
['control', 'pC1dpublished1_newstim_offvson', 'perframe_aggression']
```