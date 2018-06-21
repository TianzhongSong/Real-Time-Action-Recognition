# coding=utf8

def joint_completion(joint):
    if 8 in joint and 11 not in joint:
        joint[11] = joint[8]
    elif 8 not in joint and 11 in joint:
        joint[8] = joint[11]
    if 9 in joint and 12 not in joint:
        joint[12] = joint[9]
    elif 9 not in joint and 12 in joint:
        joint[9] = joint[12]

    return joint

def joint_filter(joint):
    if 1 not in joint:
        return False
    if 8 not in joint and 11 not in joint:
        return False
    if 9 not in joint and 12 not in joint:
        return False
    return True
