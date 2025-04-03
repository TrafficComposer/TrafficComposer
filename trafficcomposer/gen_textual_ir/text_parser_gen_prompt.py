import re
import os
import time
from tqdm import tqdm


def gen_prompt(traffic_description):
    prompts = []

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

Step 6 - Save all elements obtained above to a YAML, following the below syntax. Note that the final YAML output should start with `<YAML>` and end with `</YAML>`.
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
"""

    example_scenario_description = """
It is currently daytime with partly cloudy skies.
Our ego vehicle is driving on the third lane from the left on a five-lane road at the speed of 30 mph. 
There are pedestrians crossing at the intersection directly ahead of it. 
There is a black sedan in front of the ego vehicle in the right lane, going forward at the speed of 45 mph.
There are cars driving in front of the ego vehicle in both left two lanes, going forward at the speed of 35 mph.  
The traffic light for the direction of the ego vehicle is currently green.
"""

    example_output = """
Step 1: The `weather` attribute is explicitly specified in the textual description as `partly cloudy`. This `weather` description is not in the candidate options `[clear, cloudy, foggy, rainy, snowy]`. Therefore, we choose the most similar option, `cloudy`.
Output of Step 1: `weather: cloudy`

Step 2: The `time` attribute is explicitly specified in the textual description, being `daytime`. This `time` description is in the candidate options `[daytime, nighttime]`. Therefore, we output `daytime`.
Output of Step 2: `time: daytime`

Step 3: The `road_network` attribute is derived from the textual description. The `road_type` is `intersection`. The `traffic_sign` is not explicitly mentioned, hence `None`. The `traffic_light` is `green`. And the `lane_number` is `5`.
Output of Step 3:
road_network: 
    road_type: intersection
    traffic_sign: None
    traffic_light: green
    lane_number: 5

Step 4: The ego vehicle's attributes are derived from the textual description. The `current_behavior` is `go forward`. The `position_target` is `intersection`. The `position_relation` is `behind`. And the `speed` is `30 mph`. The textual description mentions that the intersection is directly ahead of the ego vehicle, so the ego vehicle is `behind` the intersection.
Output of Step 4:
ego_vehicle:
    current_behavior: go forward
    position_target: intersection
    position_relation: behind
    speed: 30 mph

Step 5: The other actors' attributes are derived from the textual description. The `type` of `other_actor_1` is `pedestrian`; the `current_behavior` is `crossing`; the `position_target` is `ego_vehicle`; the `position_relation` is `front`; and the `speed` is `None`. The `other_actor_2` is the "black sedan" in the textual description. Its `type` is `car`; the `current_behavior` is `go forward`; the `position_target` is `ego_vehicle`; the `position_relation` is `right front`; and the `speed` is `45 mph`. The `type` of `other_actor_3` is `car`; the `current_behavior` is `go forward`; the `position_target` is `ego_vehicle`; the `position_relation` is `left front`; and the `speed` is `35 mph`.
Output of Step 5:
participant:
    other_actor_1:
        type: pedestrian
        current_behavior: crossing
        position_target: ego_vehicle
        position_relation: front
        speed: None
    other_actor_2:
        type: car
        current_behavior: go forward
        position_target: ego vehicle
        position_relation: right front
        speed: 45 mph
    other_actor_3:
        type: car
        current_behavior: go forward
        position_target: ego vehicle
        position_relation: left front
        speed: 35 mph

Step 6: Save all elements obtained above to the following YAML format:
<YAML>
environment:
    weather: cloudy
    time: daytime
road_network: 
    road_type: intersection
    traffic_sign: None
    traffic_light: green
    lane_number: 5
participant: 
    ego_vehicle:
        current_behavior: go forward
        position_target: intersection
        position_relation: behind
        speed: 30 mph
    other_actor_1:
        type: pedestrian
        current_behavior: crossing
        position_target: intersection
        position_relation: left
        speed: None
    other_actor_2:
        type: car
        current_behavior: go forward
        position_target: ego vehicle
        position_relation: right front
        speed: 45 mph
    other_actor_3:
        type: car
        current_behavior: go forward
        position_target: ego vehicle
        position_relation: left front
        speed: 35 mph
</YAML>
"""

    example_scenario_description_2 = """
It is currently at night with clear visibility.
Our ego vehicle is driving on the second lane from the left on a four-lane road at a speed of 25 mph.
There are pedestrians crossing at the intersection directly ahead of it at the speed of 3 mph. 
There is a white sedan in front of the ego vehicle in the left lane, going forward, at the speed of 30 mph.
There are cars driving in front of the ego vehicle in both right two lanes, going forward, at the speed of 35 mph.
There is a stop sign at the intersection.
"""

    example_output_2 = """
Step 1: The `weather` attribute is explicitly specified in the textual description by "with clear visibility". This `weather` description matches the `clear` in the candidate options `[clear, cloudy, foggy, rainy, snowy]`. 
Output of Step 1: `weather: clear`

Step 2: The `time` attribute is explicitly specified in the textual description, being `night`. This `time` description is in the candidate options `[daytime, nighttime]`. Therefore, we output `nighttime`.
Output of Step 2: `time: nighttime`

Step 3: The `road_network` attribute is derived from the textual description. The `road_type` is `intersection`, the `traffic_sign` is `stop sign`, the `traffic_light` is not explicitly mentioned, and the `lane_number` is `5`.
Output of Step 3:
road_network: 
    road_type: intersection
    traffic_sign: stop sign
    traffic_light: None
    lane_number: 4

Step 4: The ego vehicle's attributes are derived from the textual description. The `current_behavior` is `go forward`, the `position_target` is `intersection`, the `position_relation` is `behind`, and the `speed` is `25 mph`. The textual description mentions that the intersection is directly ahead of the ego vehicle, so the ego vehicle is behind the intersection.
Output of Step 4:
ego_vehicle:
    current_behavior: go forward
    position_target: intersection
    position_relation: behind
    speed: 25 mph

Step 5: Other actors' attributes are derived from the textual description. The `type` of `other_actor_1` is `pedestrian`, the `current_behavior` is `crossing`, the `position_target` is `ego_vehicle`, the `position_relation` is `front`, and the `speed` is `None`. The `other_actor_2` is the "white sedan" in the textual description. Its `type` is `car`, the `current_behavior` is `go forward`, the `position_target` is `ego_vehicle`, the `position_relation` is `right front`, and the `speed` is `None`. The `type` of `other_actor_3` is `car`, the `current_behavior` is `go forward`, the `position_target` is `ego_vehicle`, the `position_relation` is `left front`, and the `speed` is `None`.
Output of Step 5:
participant:
    other_actor_1:
        type: pedestrian
        current_behavior: crossing
        position_target: ego_vehicle
        position_relation: front
        speed: 3 mph
    other_actor_2:
        type: car
        current_behavior: go forward
        position_target: ego vehicle
        position_relation: left front
        speed: 30 mph
    other_actor_3:
        type: car
        current_behavior: go forward
        position_target: ego vehicle
        position_relation: right front
        speed: 35 mph


Step 6: Save all elements obtained above to the following YAML format:
<YAML>
environment:
    weather: clear
    time: nighttime
road_network: 
    road_type: intersection
    traffic_sign: stop sign
    traffic_light: None
    lane_number: 4
participant: 
    ego_vehicle:
        current_behavior: go forward
        position_target: intersection
        position_relation: behind
        speed: 25 mph
    other_actor_1:
        type: pedestrian
        current_behavior: crossing
        position_target: ego_vehicle
        position_relation: front
        speed: 3 mph
    other_actor_2:
        type: car
        current_behavior: go forward
        position_target: ego vehicle
        position_relation: left front
        speed: 30 mph
    other_actor_3:
        type: car
        current_behavior: go forward
        position_target: ego vehicle
        position_relation: right front
        speed: 35 mph
</YAML>
"""

    prompts.append({"role": "system", "content": system_instruction})
    prompts.append({"role": "user", "content": example_scenario_description})
    prompts.append({"role": "assistant", "content": example_output})
    prompts.append({"role": "user", "content": example_scenario_description_2})
    prompts.append({"role": "assistant", "content": example_output_2})
    prompts.append({"role": "user", "content": traffic_description})

    return prompts
