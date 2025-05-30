import os
import dynesty

root_dir = r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\EMCCD only\david\CAP\CAP_2frag_but_1"  # Replace this

for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".dynesty"):
            filepath = os.path.join(dirpath, file)
            print(f"Processing: {filepath}")
            try:
                res = dynesty.DynamicNestedSampler.restore(filepath)
                res.rstate = None
                res.sampler.rstate = None
                res.save(filepath)  # Overwrite same file
            except Exception as e:
                print(f"Failed to process {filepath}: {e}")
