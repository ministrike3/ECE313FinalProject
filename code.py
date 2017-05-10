import scipy.io
import numpy as np
def data_import(patientlist):
    actual_data=[]
    for patient in patientlist:
        data = scipy.io.loadmat(patient)
        building_data = data['all_data']
        length = len(building_data[0])
        building_data = np.swapaxes(building_data, 0, 1)
        building_data = np.floor(building_data)

        full_data_as_list = np.ndarray.tolist(building_data)
        all_labels = data['all_labels']
        all_labels = all_labels[0]

        for x in range(0, len(all_labels)):
            full_data_as_list[x].append(all_labels[x])
        actual_data.append(full_data_as_list)
    return(actual_data)
def find_min(array1234,y):
    min=0
    for x in array1234:
        current=x[y]
        if current<min:
            min=current
    return(int(min))
def find_max(array1234,y):
    max=0
    for x in array1234:
        current=x[y]
        if current>max:
            max=current
    return(int(max))
def prior_probabilities(training_data):
    h1_testing=0
    for x in range(0,len(training_data)):
        h1_testing+=training_data[x][7]
    h1_testing=h1_testing/len(training_data)
    h0_testing=1-h1_testing
    return(h0_testing,h1_testing)
def create_training_testing(full_data_as_list):
    training_data = full_data_as_list[0:int(2 * len(full_data_as_list) / 3)]
    testing_data = full_data_as_list[int(2 * len(full_data_as_list) / 3):]
    training_data_h0 = []
    training_data_h1 = []
    for x in training_data:
        if x[7] == 1:
            training_data_h1.append(x)
        else:
            training_data_h0.append(x)
    for x in testing_data:
        x.append(0)
        x.append(0)
        x.append(0)
        x.append(0)
    return (training_data, training_data_h0, training_data_h1, testing_data)
def printerror123(errorarray,feature_number1,feature_number2):
    print('For feature '+str(feature_number1+1)+","+str(feature_number2+1)+":")
    print('      %-14s   %-14s   %-14s'%('Error','False Alarm','Miss Detection'))
    print("%s   %01.12f   %01.12f   %01.12f" %('MAP', errorarray[0][0], errorarray[0][1], errorarray[0][2]))
    print("%s   %01.12f   %01.12f   %01.12f" %('ML ', errorarray[1][0], errorarray[1][1], errorarray[1][2]))
    print('')
def printerror(errorarray,feature_number):
    print('For feature '+str(feature_number+1)+":")
    print('      %-14s   %-14s   %-14s'%('Error','False Alarm','Miss Detection'))
    print("%s   %01.12f   %01.12f   %01.12f" %('MAP', errorarray[0][0], errorarray[0][1], errorarray[0][2]))
    print("%s   %01.12f   %01.12f   %01.12f" %('ML ', errorarray[1][0], errorarray[1][1], errorarray[1][2]))
    print('')
def create_likelyhoodmatrix(training_data,feature):
    storage_for_training_h0 = [0 for x in range(find_min(training_data, feature), find_max(training_data, feature) + 1)]
    storage_for_training_h1 = [0 for x in range(find_min(training_data, feature), find_max(training_data, feature) + 1)]
    # Get the number of h0's that have that value
    for x in training_data_h0:
        value = x[feature]
        value = int(value + abs(find_min(training_data, feature)))
        storage_for_training_h0[value] += 1

    # Get the number of h1's that have that value
    for x in training_data_h1:
        value = x[feature]
        value = int(value + abs(find_min(training_data, feature)))
        storage_for_training_h1[value] += 1

    newarray = []
    appendable = []
    for x in range(0, abs(find_min(training_data, feature)) + find_max(training_data, feature) + 1):
        appendable.append(x - abs(find_min(training_data, feature)))
        appendable.append(storage_for_training_h0[x])
        appendable.append(storage_for_training_h1[x])
        newarray.append(appendable)
        appendable = []
    probabilities=prior_probabilities(training_data)
    for x in newarray:
        x[1] = x[1] / sum(storage_for_training_h0)
        x[2] = x[2] / sum(storage_for_training_h1)
        if x[2]>=x[1]:
            x.append(1)
        else:
            x.append(0)
        if x[2]*probabilities[1]>=x[1]*probabilities[0]:
            x.append(1)
        else:
            x.append(0)
    return(newarray)
def ML_MAP_generation(feature,testing_data,feature_number):
    testing_no_alarm = 0
    testing_alarm = 0
    for row in testing_data:
        value = row[feature_number]
        for training_value in feature:
            if value == training_value[0]:
                ML_value = training_value[3]
                MAP_value = training_value[4]
                row[8]=(ML_value)
                row[9]=(MAP_value)
        golden_alarm = row[7]
        if golden_alarm == 1:
            testing_alarm += 1
        else:
            testing_no_alarm += 1
    prob=prior_probabilities(training_data)
    false_alarm_MAP = 0
    false_alarm_ML = 0
    miss_detection_MAP = 0
    miss_detection_ML = 0
    for row in testing_data:
        golden_alarm = row[7]
        ML_prediction = row[8]
        MAP_prediction = row[9]
        if golden_alarm == 1:
            if ML_prediction == 0:
                miss_detection_ML += 1
            if MAP_prediction == 0:
                miss_detection_MAP += 1
        if golden_alarm == 0:
            if ML_prediction == 1:
                false_alarm_ML += 1
            if MAP_prediction == 1:
                false_alarm_MAP += 1
    false_alarm_MAP = false_alarm_MAP / testing_no_alarm
    false_alarm_ML = false_alarm_ML / testing_no_alarm
    miss_detection_MAP = miss_detection_MAP / testing_alarm
    miss_detection_ML = miss_detection_ML / testing_alarm
    MAP_Error = prob[0]*false_alarm_MAP + prob[1]*miss_detection_MAP
    ML_Error = (false_alarm_ML * testing_no_alarm + miss_detection_ML * testing_alarm) / (
    testing_no_alarm + testing_alarm)
    return([[MAP_Error,false_alarm_MAP,miss_detection_MAP],[ML_Error,false_alarm_ML,miss_detection_ML]])
def create_likelyhoodmatrix123(training_data,feature1,feature2):
    array_of_size_of_second_feature_h0 = []
    array_of_size_of_first_feature_h0 = []
    for x in range(find_min(training_data, feature1), find_max(training_data, feature1) + 1):
        array_of_size_of_second_feature_h0 = [0 for x in range(find_min(training_data, feature2),
                                                            find_max(training_data, feature2) + 1)]
        array_of_size_of_first_feature_h0.append(array_of_size_of_second_feature_h0)
        array_of_size_of_second_feature_h0 = []
    array_of_size_of_second_feature_h1 = []
    array_of_size_of_first_feature_h1 = []
    for x in range(find_min(training_data, feature1), find_max(training_data, feature1) + 1):
        array_of_size_of_second_feature_h1 = [0 for x in range(find_min(training_data, feature2),
                                                            find_max(training_data, feature2) + 1)]
        array_of_size_of_first_feature_h1.append(array_of_size_of_second_feature_h1)
        array_of_size_of_second_feature_h1 = []
    h0=0
    for x in training_data_h0:
        i = x[feature1]
        i = int(i + abs(find_min(training_data, feature1)))
        j = x[feature2]
        j = int(j + abs(find_min(training_data, feature2)))
        array_of_size_of_first_feature_h0[i][j]+=1
        h0+=1
    h1=0
    for x in training_data_h1:
        i = x[feature1]
        i = int(i + abs(find_min(training_data, feature1)))
        j = x[feature2]
        j = int(j + abs(find_min(training_data, feature2)))
        array_of_size_of_first_feature_h1[i][j]+=1
        h1+=1
    final=[]
    appendable=[]
    probabilities=prior_probabilities(training_data)
    for i in range(find_min(training_data, feature1), find_max(training_data, feature1) + 1):
        for j in range(find_min(training_data, feature2), find_max(training_data, feature2) + 1):
            appendable = []
            appendable.append(int(i + abs(find_min(training_data, feature1))))
            appendable.append(int(j + abs(find_min(training_data, feature1))))
            appendable.append((array_of_size_of_first_feature_h0[i][j])/(h0))
            appendable.append((array_of_size_of_first_feature_h1[i][j])/h1)
            if appendable[3] >= appendable[2]:
                appendable.append(1)
            else:
                appendable.append(0)
            if appendable[3] * probabilities[1] >= appendable[2] * probabilities[0]:
                appendable.append(1)
            else:
                appendable.append(0)
            final.append(appendable)
    #for row in final:
        #print(row)
    return (final)
def ML_MAP_generation123(feature,testing_data,feature_number1,feature_number2):
    testing_no_alarm = 0
    testing_alarm = 0
    for row in testing_data:
        i = row[feature_number1]
        j = row[feature_number2]
        for training_value in feature:
            if i == training_value[0]:
                if j== training_value[1]:
                    row[10]=training_value[4]
                    row[11]=training_value[5]
        golden_alarm = row[7]
        if golden_alarm == 1:
            testing_alarm += 1
        else:
            testing_no_alarm += 1

    false_alarm_MAP = 0
    false_alarm_ML = 0
    miss_detection_MAP = 0
    miss_detection_ML = 0
    for row in testing_data:
        golden_alarm = row[7]
        ML_prediction = row[10]
        MAP_prediction = row[11]
        if golden_alarm == 1:
            if ML_prediction == 0:
                miss_detection_ML += 1
            if MAP_prediction == 0:
                miss_detection_MAP += 1
        if golden_alarm == 0:
            if ML_prediction == 1:
                false_alarm_ML += 1
            if MAP_prediction == 1:
                false_alarm_MAP += 1
    false_alarm_MAP = false_alarm_MAP / testing_no_alarm
    false_alarm_ML = false_alarm_ML / testing_no_alarm
    miss_detection_MAP = miss_detection_MAP / testing_alarm
    miss_detection_ML = miss_detection_ML / testing_alarm
    MAP_Error = prob[0]*false_alarm_MAP + prob[1]*miss_detection_MAP
    ML_Error = (false_alarm_ML*testing_no_alarm + miss_detection_ML*testing_alarm)/(testing_no_alarm+testing_alarm)
    return([[MAP_Error,false_alarm_MAP,miss_detection_MAP],[ML_Error,false_alarm_ML,miss_detection_ML]])
def minMaperror(ML_MAP_error_array,featurenumber):
    first=1
    second=1
    third=1
    fourth=1
    patient1=0
    patient2=0
    patient3=0
    patient4=0
    x=1
    for patient in ML_MAP_error_array:
        feature=patient[featurenumber]
        MAP123=feature[0][0]
        if MAP123<first:
            fourth=third
            third=second
            second=first
            first=MAP123
            patient4 = patient3
            patient3 = patient2
            patient2 = patient1
            patient1 = x
        elif MAP123<second:
            fourth=third
            third=second
            second=MAP123
            patient4 = patient3
            patient3 = patient2
            patient2 = x
        elif MAP123<third:
            fourth=third
            third=MAP123
            patient4 = patient3
            patient3 = x
        elif MAP123<fourth:
            fourth=x
            patient4 = x
        x+=1
    print(patient1,patient2,patient3,patient4)
def minMlerror(ML_MAP_error_array,featurenumber):
    first=1
    second=1
    third=1
    fourth=1
    patient1=0
    patient2=0
    patient3=0
    patient4=0
    x=1
    for patient in ML_MAP_error_array:
        feature=patient[featurenumber]
        MAP123=feature[1][0]
        if MAP123<first:
            fourth=third
            third=second
            second=first
            first=MAP123
            patient4 = patient3
            patient3 = patient2
            patient2 = patient1
            patient1 = x
        elif MAP123<second:
            fourth=third
            third=second
            second=MAP123
            patient4 = patient3
            patient3 = patient2
            patient2 = x
        elif MAP123<third:
            fourth=third
            third=MAP123
            patient4 = patient3
            patient3 = x
        elif MAP123<fourth:
            fourth=x
            patient4 = x
        x+=1
    print(patient1,patient2,patient3,patient4)
def list_features_increasing_MAP_order(ML_MAP_error_array,patientnumber):
    set123=ML_MAP_error_array[patientnumber]
    first=1
    second=1
    third=1
    fourth=1
    patient1=0
    patient2=0
    patient3=0
    patient4=0
    x=1
    for feature in set123:
        MAP123=feature[0][0]
        if MAP123<first:
            fourth=third
            third=second
            second=first
            first=MAP123
            patient4 = patient3
            patient3 = patient2
            patient2 = patient1
            patient1 = x
        elif MAP123<second:
            fourth=third
            third=second
            second=MAP123
            patient4 = patient3
            patient3 = patient2
            patient2 = x
        elif MAP123<third:
            fourth=third
            third=MAP123
            patient4 = patient3
            patient3 = x
        elif MAP123<fourth:
            fourth=x
            patient4 = x
        x+=1
    print(patient1,patient2,patient3,patient4)
def list_features_increasing_Ml_order(ML_MAP_error_array,patientnumber):
    set123=ML_MAP_error_array[patientnumber]
    first=1
    second=1
    third=1
    fourth=1
    patient1=0
    patient2=0
    patient3=0
    patient4=0
    x=1
    for feature in set123:
        MAP123=feature[1][0]
        if MAP123<first:
            fourth=third
            third=second
            second=first
            first=MAP123
            patient4 = patient3
            patient3 = patient2
            patient2 = patient1
            patient1 = x
        elif MAP123<second:
            fourth=third
            third=second
            second=MAP123
            patient4 = patient3
            patient3 = patient2
            patient2 = x
        elif MAP123<third:
            fourth=third
            third=MAP123
            patient4 = patient3
            patient3 = x
        elif MAP123<fourth:
            fourth=x
            patient4 = x
        x+=1
    print(patient1,patient2,patient3,patient4)

readarray=["1_a41178.mat","2_a42126.mat","3_a40076.mat","4_a40050.mat","5_a41287.mat","6_a41846.mat","7_a41846.mat","8_a42008.mat","9_a41846.mat"]
actual_data=data_import(readarray)
ML_MAP_error_array=[]
for number in range(0,9):
    print("For Patient "+str(number+1))
    patient=actual_data[number]
    training_data, training_data_h0,training_data_h1,testing_data = create_training_testing(patient)
    prob=prior_probabilities(training_data)
    print("The prior probability of H0 is " + str(prob[0]))
    print("The prior probability of H1 is "+str(prob[1]))
    print('')
    patient_ML_MAP_array=[]
    for feature_number in range(0,7):
        feature=create_likelyhoodmatrix(training_data,feature_number)
        error_array=ML_MAP_generation(testing_data,feature_number)
        patient_ML_MAP_array.append(error_array)
        printerror(error_array,feature_number)
    ML_MAP_error_array.append(patient_ML_MAP_array)

    feature=create_likelyhoodmatrix123(training_data,0,6)
    error_array123=ML_MAP_generation123(testing_data,0,6)
    printerror123(error_array123,0,6)



print("_________________________________________________________")
print("The 5 patients with the smallest MAP errors for each feature are:")
for number in range(0,7):
    print('for feature '+str(number+1))
    minMaperror(ML_MAP_error_array,number)
print("_________________________________________________________")
print("The 5 patients with the smallest ML errors for each feature are:")
for number in range(0,7):
    print('for feature '+str(number+1))
    minMlerror(ML_MAP_error_array,number)
print("_________________________________________________________")
print("The 5 features with the smallest MAP error are:")
for number in range(0,9):
    print('for patient '+str(number+1))
    list_features_increasing_MAP_order(ML_MAP_error_array,number)
print("_________________________________________________________")
print("The 5 features with the smallest ML error are:")
for number in range(0,9):
    print('for patient '+str(number+1))
    list_features_increasing_Ml_order(ML_MAP_error_array,number)

ML=[]
MAP=[]
for number in [1,4,6]:
    print("For Patient "+str(number+1))
    patient=actual_data[number]
    training_data, training_data_h0,training_data_h1,testing_data = create_training_testing(patient)
    feature=create_likelyhoodmatrix123(training_data,0,6)
    error_array123=ML_MAP_generation123(feature,testing_data,0,6)
    ML.append(error_array123[1][0])
    MAP.append(error_array123[0][0])
    printerror123(error_array123,0,6)


x=0
y=0
for item in ML:
    x=x+item
for item in MAP:
    y=y+item
x=x/len(ML)
y=y/len(MAP)
print("The Average MAP error is " + str(y))
print("The Average ML error is  " + str(x))
for number in range(0,9):
    print("For Patient "+str(number+1))
    patient=actual_data[number]
    training_data, training_data_h0,training_data_h1,testing_data = create_training_testing(patient)
    feature=create_likelyhoodmatrix123(training_data,0,6)
    error_array123=ML_MAP_generation123(feature,testing_data,0,6)
    printerror123(error_array123,0,6)