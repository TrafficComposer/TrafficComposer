import os 


def is_image_file(filename):
    filename = filename.lower()
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def gen_img_list(img_dir_path, fp_save=None, debug=False):
    """
    Generate a list of image files in the directory img_dir_path
    return:
        ls_ans: list of image file paths
            [img_file_path_0, img_file_path_1, ...]
    """
    assert os.path.isdir(img_dir_path), f"{img_dir_path} is not a directory."
    ls_img_dir = os.listdir(img_dir_path)
    ls_img_dir.sort()

    ls_ans = []
    for img_fn in ls_img_dir:
        full_path = os.path.join(img_dir_path, img_fn)
        if debug:
            print(full_path, end=" ")
        if is_image_file(img_fn):
            ls_ans.append(full_path)
            if debug:
                print("added.")
        else:
            if debug:
                print("not image, discarded.")

    if debug:
        print(ls_ans)

    if fp_save is not None:
        with open(fp_save, "w") as f:
            for img_fn in ls_ans:
                f.write(img_fn + "\n")
        print(f"Saved to {fp_save}")

    return ls_ans


def copy_dir(source_dir, target_dir):
    """Copy the all the contents from source_dir to target_dir."""
    if os.path.exists(target_dir):
        input(
            f"WARNING: {target_dir} already exists. Press Enter to remove it."
        )
        os.system(f"rm -r {target_dir}")
    os.system(f"cp -r {source_dir} {target_dir}")
    print(f"Results are saved to {target_dir}, copied from {source_dir}.")
