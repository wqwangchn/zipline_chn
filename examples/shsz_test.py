#!/usr/bin/env python
#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from zipline.api import symbol, record,order_target,set_benchmark

def initialize(context):
    context.sym = symbol('000001')
    context.i = 0
    #set_benchmark()

def handle_data(context, data):
    # Skip first 300 days to get full windows
    context.i += 1
    if context.i == 1:
        order_target(context.sym, 100000)

    # Compute averages
    # history() has to be called with the same params
    # from above and returns a pandas dataframe.
    short_mavg = data.history(context.sym, 'price', 2, '1d').mean()
    long_mavg = data.history(context.sym, 'price', 3, '1d').mean()

    # Trading logic
    '''if short_mavg > long_mavg:
        # order_target orders as many shares as needed to
        # achieve the desired number of shares.
        order_target(context.sym, 100000)
    elif short_mavg < long_mavg:
        order_target(context.sym, 0)'''

    # Save values for later inspection
    record(Price=data.current(context.sym, "price"),
           short_mavg=short_mavg,
           long_mavg=long_mavg)