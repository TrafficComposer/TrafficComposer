import base64
import os


from trafficcomposer.baseline.multi_modal_gpt.config_baseline_gpt4o import TRAIN_IMAGE_EG1_PATH, TRAIN_IMAGE_EG2_PATH


def encode_image(image_path):
    """
    Function to encode the image
    """
    assert os.path.exists(image_path), f"Image file not found: {image_path}"
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


system_instruction = """
You will work as an autonomous driving system testing engineer. Your task is to derive a test traffic scenario from the given traffic scenario description.

Approach this task step-by-step, take your time, and do not skip steps.

Step 1: Derive the `weather` element of the test scenario. The weather could be one option in `[clear, cloudy, foggy, rainy, snowy]`. Output the weather element. If the weather is not explicitly mentioned in the test scenario description, you can set it as the default weather `clear`. If no options can describe the weather mentioned in the test scenario description, choose the most similar one in the previously defined weather options.

Step 2: Derive the `time` of the test scenario. The time could be one option in `[daytime, nighttime]`. Output the time element. If the time is not explicitly mentioned in the test scenario description, you can set it as the default time `daytime`. If no options can describe the time mentioned in the test scenario description, choose the most similar one in the previously defined time options.

Step 3: Derive the test scenario's `road_network` element. The `road_network` element contains `road_type`, `traffic_sign`, `traffic_light` and `lane_number`. The `road_type` could be one option of `[intersection, t-intersection, roundabout]`. The `traffic_light` could be one of `[green, red]`. The `traffic_sign` could be one of `[stop sign, speed limit sign]`. For the `lane_number`, output the number of lanes in the road. If the `lane_number` is not explicitly mentioned in the test scenario description, output `None`. 
For `road_type` and `traffic_light`, if no options can describe the `road_type` and `traffic_light` mentioned in the test scenario description, choose the most similar one in the previously defined options. If no options can describe the `traffic_sign` mentioned in the test scenario description, create a new option yourself. Mention that `traffic_light` and `traffic_sign` are not always present in the test scenario description and do not usually appear in the same test scenario description. 

Step 4: Derive the `current_behavior`, `position_target`, `position_relation`, and `speed` for the ego vehicle. The `current_behavior` could be one option in `[go forward, yield, stop, turn left, turn right, change lane to left, change lane to right]`. The `position_target` is one of the reference road network elements obtained in Step 2, and `position_relation` describes the relative position between the ego vehicle and the reference position target. The options are `[front, behind, left, right, opposite, left behind, right behind, left front, right front]`. For example, if the `position target` is `intersection` and the `position relation` is `behind`, it means the ego vehicle is behind (approaching) the intersection. The `speed` could be in the format of `speed: 30 mph` or `speed: 30 km/h`. If the speed is not indicated, output `None`. If the behavior of the ego vehicle is not mentioned in the test scenario description, set it as `go forward`. If no options can describe any attribute of the ego vehicle in the test scenario description, choose the most similar one in the previously defined options. Organize the information in a YAML format, with `ego_vehicle` as the key and the attributes as the values. The YAML format is as follows:
ego_vehicle:
    current_behavior:
    position_target:
    position_relation:
    speed:

Step 5: Identify all actors other than the ego vehicle. For each actor, derive the following attributes: `type`, `current_behavior`, `position_target`, `position_relation`, and `speed`. The options of `type` of other actors could be in `[car, truck, pedestrian, bus, cyclist]`. Derive the `current_behavior`, `position_target`, `position_relation`, and `speed` of each other actor as what you did for the ego vehicle in Step 4. Note that the `position_target` of other actors can be `ego vehicle`, one of the other actors, or one of the obtained road network elements based on the description in the test scenario description. Organize the information in a YAML format, with `other_actor_1`, `other_actor_2`, etc., as the keys and the attributes as the values. The YAML format is as follows:
other_actor_1:
    type:
    current_behavior:
    position_target:
    position_relation:
    speed:
other_actor_2:
    type:
    current_behavior:
    position_target:
    position_relation:
    speed:
...

Step 6: Detect the lane dividing lines in the image's road network. 

Step 7: Detect all the traffic participants in the image, including the ego vehicle and other actors. Other actors could be cars, trucks, cyclists, or pedestrians.

Step 8: For each vehicle (including the ego vehicle), determine the lane index and fill in the entity `lane_idx` of the agent vehicle. If the ego vehicle is not in the lane, output "None". The `lane_idx` of the leftmost lane is 0, and the `lane_idx` of the rightmost lane is the total number of lanes minus 1. The `lane_idx` should be exactly an integer.

Step 9: Save all elements obtained above to a YAML, following the below syntax. Note that the final YAML output should start with `<YAML>` and end with `</YAML>`.
<YAML>
environment:
    weather:
    time:
road_network:
    road_type:
    traffic_sign:
    traffic_light:
    lane_number:
participant:
    ego_vehicle:
        current_behavior:
        position_target:
        position_relation:
        speed:
    other_actor_1:
        type:
        current_behavior:
        position_target:
        position_relation:
        speed:
    ...
</YAML>
Answer the question with only the YAML format, starting with the keyword <YAML> and ending with </YAML>.
"""

train_description_eg1 = """
It is currently nighttime with clear skies.
Our ego vehicle is driving on the right-most lane on a three-lane road at 55 mph. 
A black sedan is right in front of the ego vehicle, going forward at 60 mph.
There is a white sedan driving in front of the ego vehicle in the left lane, going forward at the speed of 50 mph.
There is no traffic light or traffic sign in the scenario.
"""
base64_train_image_eg1 = encode_image(TRAIN_IMAGE_EG1_PATH)
train_aligned_gt_eg1 = """
Step 1: The `weather` element of the test scenario is explictly mentioned as `clear`, which matches the predefined options. 
Output of Step 1: `weather: clear`

Step 2: The `time` of the test scenario is explictly mentioned as `nighttime`, which matches the predefined options.
Output of Step 2: `time: nighttime`

Step 3: The `road_network` element of the test scenario is not explicitly mentioned. The `road_type` is not explicitly mentioned. The `traffic_sign` and `traffic_light` are not present in the scenario. The `lane_number` is explicitly mentioned as `3`.
Output of Step 3:
road_network:
    road_type: None
    traffic_sign: None
    traffic_light: None
    lane_number: 3

Step 4: The ego vehicle's `current_behavior` is not explicitly mentioned. The `position_target` and `position_relation` are not explicitly mentioned. The `speed` is explicitly mentioned as `55 mph`.
Output of Step 4:
ego_vehicle:
    current_behavior: go forward
    position_target: None
    position_relation: None
    speed: 55 mph

Step 5: There are two other actors in the scenario. The `type` of both actors is `car`. The `current_behavior` of both actors is `go forward`. The `position_target` of the first actor is the `ego_vehicle`, and the `position_relation` is `front`. The `speed` of the first actor is `60 mph`. The `position_target` of the second actor is the `ego_vehicle`, and the `position_relation` is `left front`. The `speed` of the second actor is `50 mph`.
Output of Step 5:
other_actor_1:
    type: car
    current_behavior: go forward
    position_target: ego_vehicle
    position_relation: front
    speed: 60 mph
other_actor_2:
    type: car
    current_behavior: go forward
    position_target: ego_vehicle
    position_relation: left front
    speed: 50 mph

Combining the information extracted from the reference image and follow Step 6, Step 7, and Step 8, we can know the `lane_idx` of the `ego_vehicle` is `2`; the `lane_idx` of `other_actor_1` is `2`; the `lane_idx` of `other_actor_2` is `1`.

Step 9: Save all elements obtained above to a YAML.
<YAML>
environment:
    weather: clear
    time: nighttime
road_network:
    road_type: None
    traffic_sign: None
    traffic_light: None
    lane_number: 3
participant:
    ego_vehicle:
        current_behavior: go forward
        lane_idx: 2
        position_relation: None
        position_target: None
        speed: 55 mph
    other_actor_1:
        current_behavior: go forward
        lane_idx: 2
        position_relation: front
        position_target: ego_vehicle
        speed: 60 mph
        type: car
    other_actor_2:
        current_behavior: go forward
        lane_idx: 1
        position_relation: left front
        position_target: ego_vehicle
        speed: 50 mph
        type: car
</YAML>
"""

train_description_eg2 = """
It is currently nighttime with clear visibility.
Our ego vehicle is driving on the left lane of a two-lane road at 15 mph.
The ego vehicle is going forward and approaching an intersection.
One pedestrian is walking on the right of the ego vehicle.
A white sedan is in front of the ego vehicle in the right lane, stopping and waiting for the red traffic light.
A silver sedan is driving in front of the ego vehicle on the other side of the intersection in both left two lanes, going forward at 25 mph.
The traffic light in the direction of the ego vehicle is currently red.
"""
base64_train_image_eg2 = encode_image(TRAIN_IMAGE_EG2_PATH)

train_aligned_gt_eg2 = """
Step 1: The `weather` element of the test scenario is explictly mentioned as `clear`, which matches the predefined options.
Output of Step 1: `weather: clear`

Step 2: The `time` of the test scenario is explictly mentioned as `nighttime`, which matches the predefined options.
Output of Step 2: `time: nighttime`

Step 3: The `road_type` is explicitly mentioned as `intersection`. The `traffic_sign` is not present in the scenario. The `traffic_light` is explicitly mentioned as `red`. The `lane_number` is explicitly mentioned as `2`.
Output of Step 3:
road_network:
    road_type: intersection
    traffic_sign: None
    traffic_light: red
    lane_number: 2

Step 4: The ego vehicle's `current_behavior` is explicitly mentioned as `go forward`. The `position_target` is explicitly mentioned as `intersection`. The ego vehicle is referred to as "approaching the intersection", from which we can deduct the `position_relation` is `behind` the `intersection`. The `speed` is explicitly mentioned as `15 mph`.
Output of Step 4:
ego_vehicle:
    current_behavior: go forward
    position_target: intersection
    position_relation: behind
    speed: 15 mph

Step 5: There are three other actors in the scenario. The `type` of the first actor is `pedestrian`. The `current_behavior` of the first actor is referred to as walking, which matches the `go forward` in the candidate options. The `position_target` of the first actor is the `ego_vehicle`, and the `position_relation` is `right front`. The `speed` of the first actor is `None`. The `type` of the second actor is `car`. The `current_behavior` of the second actor is `stop`. The `position_target` of the second actor is the `ego_vehicle`, and the `position_relation` is `right front`. The `speed` of the second actor is `0 mph`. The third actor is the silver sedan. The `type` of the third actor is `car`. The `current_behavior` of the third actor is `go forward`. The `position_target` of the third actor is the `ego_vehicle`, and the `position_relation` is `front`. The `speed` of the third actor is `25 mph`.
Output of Step 5:
other_actor_1:
    type: pedestrian
    current_behavior: go forward
    position_target: ego_vehicle
    position_relation: right front
    speed: None
other_actor_2:
    type: car
    current_behavior: stop
    position_target: ego_vehicle
    position_relation: right front
    speed: 0 mph
other_actor_3:
    type: car
    current_behavior: go forward
    position_target: ego_vehicle
    position_relation: front
    speed: 25 mph

Combining the information extracted from the reference image and follow Step 6, Step 7, and Step 8, we can know the `lane_idx` of the `ego_vehicle` is `0`; the `lane_idx` of `other_actor_1` is `1`; the `lane_idx` of `other_actor_2` is `1`; the `lane_idx` of `other_actor_3` is `0`.

Step 9: Save all elements obtained above to a YAML.
<YAML>
environment:
    weather: clear
    time: nighttime
road_network:
    road_type: intersection
    traffic_sign: None
    traffic_light: red
    lane_number: 2
participant:
    ego_vehicle:
        current_behavior: go forward
        lane_idx: 0
        position_relation: behind
        position_target: intersection
        speed: 15 mph
    other_actor_1:
        type: pedestrian
        current_behavior: go forward
        lane_idx: 1
        position_relation: right front
        position_target: ego_vehicle
        speed: None
    other_actor_2:
        type: car
        current_behavior: stop
        lane_idx: 1
        position_relation: right front
        position_target: ego_vehicle
        speed: 0 mph
    other_actor_3:
        type: car
        current_behavior: go forward
        lane_idx: 0
        position_relation: front
        position_target: ego_vehicle
        speed: 25 mph
</YAML>
"""


def gen_multi_modal_prompt(traffic_description, image_path, detail="low"):
    base64_image = encode_image(image_path)

    prompt = [
        {
            "role": "system",
            "content": system_instruction,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": train_description_eg1},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_train_image_eg1}",
                        "detail": detail,
                    },
                },
            ],
        },
        {"role": "assistant", "content": train_aligned_gt_eg1},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": train_description_eg2},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_train_image_eg2}",
                        "detail": detail,
                    },
                },
            ],
        },
        {"role": "assistant", "content": train_aligned_gt_eg2},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": traffic_description},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": detail,
                    },
                },
            ],
        },
    ]
    return prompt
