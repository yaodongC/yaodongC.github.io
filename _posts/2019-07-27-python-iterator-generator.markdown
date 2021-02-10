---
layout: post
title: "iterator and Generator"
subtitle:   A Beginner's Guid to Python 02
date:       2019-07-20
author:     Yaodong Cui
header-img: img/pytorch-logo-dark.svg
header-mask: 0.5
catalog: true
tags:
    - Python
---


# Iterators in Python

Iterator in Python is simply an object that can be iterated upon. An object which will return data, one element at a time.

Technically speaking, a Python iterator object must implement two special methods, '__iter__()' and '__next__()', collectively called the iterator protocol.

An object is called iterable if we can get an iterator from it. Most built-in containers in Python like: list, tuple, string etc. are iterables. The for loop was able to iterate automatically through the list. In fact the for loop can iterate over any iterable.

```python
for element in my_list:
    print(element)
```
This 'for' loop is implemented as such:
```python
# create an iterator object from that iterable
iter_obj = iter(iterable)

# infinite loop
while True:
    try:
        # get the next item
        element = next(iter_obj)
        # do something with element
    except StopIteration:
        # if StopIteration is raised, break from loop
        break
```

The 'iter()' function (which in turn calls the '__iter__()' method) returns an iterator from them. If required, some initialization can be performed.

```python
class PowTwo:
    """Class to implement an iterator
    of powers of two"""

    def __init__(self, max=0):
        self.max = max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= self.max:
            result = 2 ** self.n
            self.n += 1
            return result
        else:
```

We use the 'next()' function to manually iterate through all the items of an iterator. When we reach the end and there is no more data to be returned, it will raise the 'StopIteration' Exception.



# Generators in Python

Python generators are a simple way of creating iterators. All the work we mentioned above are automatically handled by generators in Python. Simply speaking, a generator is a function that returns an object (iterator) which we can iterate over (one value at a time).

If a function contains at least one yield statement (it may contain other 'yield' or 'return' statements), it becomes a generator function. Both 'yield' and 'return' will return some value from a function. The difference is that while a 'return' statement terminates a function entirely, 'yield' statement pauses the function saving all its states and later continues from there on successive calls.

Here is how a generator function differs from a normal function:
- Generator function contains one or more 'yield' statements.
- When called, it returns an object (iterator) but does not start execution immediately.
- Methods like '__iter__()' and '__next__()' are implemented automatically. So we can iterate through the items using 'next()'.
- Once the function yields, the function is **paused** and the control is transferred to the caller.
- Local variables and their states are **remembered between successive calls**.
- Finally, when the function terminates, 'StopIteration' is raised automatically on further calls.

Unlike normal functions, the local variables are **not destroyed** when the function 'yields'. Furthermore, the generator object can be iterated only once. To restart the process we need to create another generator object.

One final thing to note is that we can use generators with for loops directly.This is because a for loop takes an iterator and iterates over it using next() function. It automatically ends when StopIteration is raised. Check here to know how a for loop is actually implemented in Python.


```python
# A simple generator function
def my_gen():
    n = 1
    print('This is printed first')
    # Generator function contains yield statements
    yield n

# Using for loop
for item in my_gen():
    print(item)
```
