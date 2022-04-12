# Franka_Analytical_IK_wrapped

### 1. Install pybind11 library

```
pip install pybind11
```

### 2. compile franka_ik_pybind.cpp


```
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) franka_ik_pybind.cpp -o franka_ik_pybind$(python3-config --extension-suffix)
```

### 3. import compiled library

```
import franka_ik_pybind
```

### 4. use functions in python

```
python surface_contact_pybind.py
```

```
import franka_ik_pybind
...
print(franka_ik_pybind.franka_IK(handPosition[i], handOrientation[i], random.uniform(-2.8973, 2.8973), initialJointPosition[:-2]))
...
```


#### Reference

https://github.com/ffall007/franka_analytical_ik

https://pybind11.readthedocs.io/en/stable/index.html
