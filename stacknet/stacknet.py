import os

import sys
sys.path.append('/home/weihao/posenet/paranet')
from network import Network


class StackNet(Network):

    def parameters(self, dim_input=2, dim_output=4, dim_ref=10):

        self.dim0 = dim_input + dim_ref
        self.out0 = 400
        self.weights0 = self.make_var('weights0', shape=[self.dim0, self.out0])
        self.biases0 = self.make_var('biases0', [self.out0])

        self.dim1 = self.out0
        self.out1 = 40
        self.weights1 = self.make_var('weights1', shape=[self.dim1, self.out1])
        self.biases1 = self.make_var('biases1', [self.out1])

        self.dim2 = self.out1
        self.out2 = dim_output
        self.weights2 = self.make_var('weights2', shape=[self.dim2, self.out2])
        self.biases2 = self.make_var('biases2', [self.out2])

        self.dim3 = self.out1
        self.out3 = dim_ref
        self.weights3 = self.make_var('weights3', shape=[self.dim3, self.out3])
        self.biases3 = self.make_var('biases3', [self.out3])

    def set_stacK(self, stack):
        self.stack = stack

    def setup(self):
        self.parameters()

        ref_out_name = 'ref0'
        for a in range(self.stack):
            input_name = 'input{}'.format(a)
            ref_name = ref_out_name
            icp_name = 'icp{}_in'.format(a)
            ifc0_name = 'ifc0{}_in'.format(a)
            ifc1_name = 'ifc1{}_in'.format(a)
            ifcr0_name = 'ifcr0{}_in'.format(a)
            ifcr1_name = 'ifcr1{}_in'.format(a)
            output_name = 'output{}'.format(a)
            (self.feed(input_name, ref_name)
             .concat(1, name=icp_name)
             .fc_w(name=ifc0_name,
                  weights=self.weights0,
                  biases=self.biases0)
             .sigmoid(name=ifcr0_name)
             .fc_w(name=ifc1_name,
                   weights=self.weights1,
                   biases=self.biases1)
             .sigmoid(name=ifcr1_name)
             .fc_w(name=output_name,
                   weights=self.weights2,
                   biases=self.biases2)
             )

            ref_fc_name = 'reffc{}'.format(a+1)
            ref_out_name = 'ref{}'.format(a+1)
            (self.feed(ifcr1_name)
             .fc_w(name=ref_fc_name,
                   weights=self.weights3,
                   biases=self.biases3)
             .sigmoid(name=ref_out_name)
            )


        print("number of layers = {}".format(len(self.layers)))


    def setup1(self):
        self.parameters()

        ref_out_name = 'ref0'
        for a in range(self.stack):
            input_name = 'input{}'.format(a)
            ref_name = ref_out_name
            icp_name = 'icp{}_in'.format(a)
            ifc0_name = 'ifc0{}_in'.format(a)
            ifc1_name = 'ifc1{}_in'.format(a)
            ifcr0_name = 'ifcr0{}_in'.format(a)
            ifcr1_name = 'ifcr1{}_in'.format(a)
            output_name = 'output{}'.format(a)
            (self.feed(input_name, ref_name)
             .concat(1, name=icp_name)
             .fc(400, name=ifc0_name)
             .sigmoid(name=ifcr0_name)
             .fc(200, name=ifc1_name,)
             .sigmoid(name=ifcr1_name)
             .fc(4, name=output_name)
             )

            ref_fc_name = 'reffc{}'.format(a+1)
            ref_out_name = 'ref{}'.format(a+1)
            (self.feed(ifcr1_name)
             .fc(10, name=ref_fc_name)
             .sigmoid(name=ref_out_name)
            )


        print("number of layers = {}".format(len(self.layers)))
