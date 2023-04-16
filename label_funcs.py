# Label extraction functions

def binary_gender_label(filename):
    # Infer gender class from filename
    # e.g.  'F_30_1234.jpg' -> 0
    #       'M_30_1234.jpg' -> 1
    label = filename.split('_')[0].upper()
    label = (0 if label == 'F'
            else 1 if label == 'M'
            else None)
    return label

def age_int_label(filename):
    # Infer age from filename
    # e.g.  'M_28_1234.jpg' -> 28
    #       'F_41_1234.jpg' -> 41
    label = int(filename.split('_')[1])
    return label

def age_float_label(filename):
    # Infer age from filename
    # e.g.  'M_28_1234.jpg' -> 28
    #       'F_41_1234.jpg' -> 41
    label = float(filename.split('_')[1])
    return label

def age_class_label(cutoffs):
    # Create unique class-getter function depending on the class_maxes
    def f(filename):
        # Get an age class from filename
        # e.g.  'M_28_1234.jpg' -> 2
        #       'F_41_1234.jpg' -> 4
        label = age_int_label(filename)
        for c, class_max in enumerate(cutoffs):
            if label <= class_max:
                return c
        return c+1
    return f


def dir_label_func(f):
    return int(f.split(' ')[0])