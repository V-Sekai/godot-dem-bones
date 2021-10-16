def can_build(env, platform):    
    return False and not env["disable_3d"]


def configure(env):
    pass

def get_doc_classes():
    return [
        "BlendShapeBake",
    ]

def get_doc_path():
    return "doc_classes"