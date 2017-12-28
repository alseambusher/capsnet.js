/*jshint esversion: 6 */
/***
License: MIT
Author: Suresh Alse
***/

import {Array3D,Array4D, NDArrayMathGPU, Scalar, NDArray} from 'deeplearn';
import cfg from './config.js';

const math = new NDArrayMathGPU();
const epsilon = Scalar.new(1e-9)

class CapsLayer {
  constructor (num_outputs, vec_len, with_routing=true, layer_type="FC", batch_size=128) {
    this.num_outputs = num_outputs;
    this.vec_len = vec_len;
    this.with_routing = with_routing;
    this.layer_type = layer_type;
    this.batch_size = batch_size;
  }

  feed(input, kernel_size=null, stride=null) {
    if (this.layer_type == "CONV") {
      this.kernel_size = kernel_size;
      this.stride = stride;

      if(!this.with_routing) {
        // the PrimaryCaps layer, a cnn layer
        console.log(input)
        console.log(input.shape, input.dtype);

        for (let i=0; i < this.batch_size; i++) {

        }
        let caps = math.relu(math.conv2d(input, this.kernel_size, null, this.stride, 'valid'));
        caps = math.reshape(caps, [128, -1, this.vec_len, 1]);
        caps = squash(caps);
        console.log(caps);
        return caps;
      }
    }


  }
}

function squash(vector) {
  let vec_squared_norm = math.sum(math.square(vector), -2, true);
  // scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
  let scalar_factor = math.divide(math.divide(vec_squared_norm, math.sum(Scalar.new(1), vec_squared_norm)), math.sqrt(math.sum(vec_squared_norm, epsilon)));
  let vec_squashed = math.multiply(scalar_factor, vector);
  return vec_squashed;
}

let i = new CapsLayer(null, 8, false, "CONV")
const shape = [2, 3];  // 2 rows, 3 columns
const a =  Array4D.new([128, 20, 20, 256], Array.from({length: 128*20*20*256}, () => Math.random()));

const filter = NDArray.randNormal([9,9,1,1]);
j = i.feed(a, filter, [1,1])
console.log(j.data().then(data => console.log(data)))
