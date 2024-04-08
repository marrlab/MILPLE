import os

# what=["gradcam", "lrp"]
# what=["gradcam", "lrp", "input_iba", "feat_iba"]
what = ["rnd","gradcam", "lrp", "input_iba", "feat_iba"]
percents = ["0.1","0.2","0.3","0.5","0.7","0.9","0.4","0.8"]
# percents = ["0.4","0.8","0.85","0.95","1.00"]

if os.path.exists("run_mds.sh"):
    os.remove("run_mds.sh")

f = open("run_mds.template", "r")
sh = open("run_mds.sh", "w")

sh.writelines("#!/bin/bash\n\n")

lines = f.readlines()

for mod in what:
    for run in percents:
        if os.path.exists("sb-" + mod + "-" + run + ".cmd"):
            os.remove("sb-" + mod + "-" + run  + ".cmd")

for mod in what:
    for run in percents:
        c = lines.copy()
        for i in range(len(c)):
            c[i] = c[i].replace("@1@", mod)
            c[i] = c[i].replace("@2@", run)

            with open("sb-" + mod + "-" + run + ".cmd", "w") as g:
                g.writelines(c)
                g.flush()
                g.close()

        sh.writelines("sbatch " + "sb-" + mod + "-" + run + ".cmd\n")

sh.flush()
sh.close()


