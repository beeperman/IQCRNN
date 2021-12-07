import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class RNNController(object):

    def __init__(self):
        self.is_bias = None
        self.activation = None

        self.ob_dim = None
        self.ac_dim = None
        self.xi_dim = None
        self.hid_dim = None

        self.ph_input_nto = None
        self.ph_ac_nta = None
        self.ph_adv_nt = None
        self.ph_dn_nt = None
        self.ph_initstate_s = None
        self.sy_output_nta = None

        self.ph1_input_no = None
        self.ph1_initstate_ns = None
        self.sy1_output_na = None
        self.sy1_nextstate_ns = None

        # phi info
        self.Aphi = None
        self.Bphi = None

        # weights
        self.AK = None
        self.BK1= None
        self.BK2= None
        self.CK1= None
        self.DK1= None
        self.DK2= None
        self.CK2= None
        self.DK3= None

        self.bxi= None
        self.bu = None
        self.bv = None

        # assign ops
        self.assign_AK = None
        self.assign_BK1= None
        self.assign_BK2= None
        self.assign_CK1= None
        self.assign_DK1= None
        self.assign_DK2= None
        self.assign_CK2= None
        self.assign_DK3= None

        # phs for assignment
        self.aph_AK = None
        self.aph_BK1= None
        self.aph_BK2= None
        self.aph_CK1= None
        self.aph_DK1= None
        self.aph_DK2= None
        self.aph_CK2= None
        self.aph_DK3= None


    def build_rnn(
            self,
            initstate_placeholder,
            input_size,
            step_num,
            output_size,
            scope,
            hid_nodes_size,
            states_size,
            Aphi,
            Bphi,
            activation=tf.nn.tanh,
            is_bias=False,
            output_activation=None):
        """
            Builds a feedforward neural network
            arguments:
                input_placeholder: placeholder variable for the state (batch_size, input_size)
                output_size: size of the output layer
                scope: variable scope of the network
                n_layers: number of hidden layers
                hid_nodes_size: dimension of the hidden layer
                activation: activation of the hidden layers
                output_activation: activation of the ouput layers

            returns:
                output placeholder of the network (the result of a forward pass)
        """
        # raise NotImplementedError

        with tf.variable_scope(scope):
            #batch_size = input_placeholder.shape[0]

            if initstate_placeholder is None:  # TODO: if none then 0 else repleat to the match size
                #sy_initstate = tf.zeros(tf.stack([batch_size, states_size]))
                sy_initstate = tf.zeros(tf.stack([1, states_size]))
            else: # broadcast shape if needed
                #sy_initstate = initstate_placeholder.reshape([1,-1]) + tf.zeros(tf.stack([batch_size, states_size]))
                #sy_initstate = tf.broadcast_to(initstate_placeholder.reshape([1,-1]), tf.stack([batch_size, states_size]))
                sy_initstate = tf.reshape(initstate_placeholder, [1,-1])
                pass

            input_placeholder = tf.placeholder(shape=[None, step_num, input_size], name="obrnn", dtype=tf.float32)
            termination_placeholder = tf.placeholder(shape=[None, step_num], name="dnrnn", dtype=tf.float32)

            sy_input = tf.unstack(input_placeholder, axis=1) # input: nto
            sy_done = tf.unstack(tf.reshape(termination_placeholder, [-1, step_num, 1]), axis=1) # input: nt

            new_activation = self.build_tilde_activation(activation, Aphi, Bphi)

            #  xi(k+1) = AK  xi(k) + BK1 (B-A/2) z(k) + BK2 y(k)
            #  u(k)    = CK1 xi(k) + DK1 (B-A/2) z(k) + DK2 y(k)
            #  v(k)    = CK2 xi(k) + DK3 y(k)
            #  z(k)    = phi~(v(k))
            #
            #  xi: hidden state
            #  y:  input
            #  v:  after dense
            #  w:  after activation
            #  u:  output
            #  TODO: biased case? 1 in y?

            # xi
            AK = tf.Variable(tf.random.truncated_normal(shape=[states_size, states_size], stddev=0.0001))
            BK1= tf.Variable(tf.random.truncated_normal(shape=[states_size, hid_nodes_size], stddev=0.0001))
            BK2= tf.Variable(tf.random.truncated_normal(shape=[states_size, input_size], stddev=0.0001))

            # u
            CK1= tf.Variable(tf.random.truncated_normal(shape=[output_size, states_size], stddev=0.0001))
            DK1= tf.Variable(tf.random.truncated_normal(shape=[output_size, hid_nodes_size], stddev=0.0001))
            DK2= tf.Variable(tf.random.truncated_normal(shape=[output_size, input_size], stddev=0.0001))

            # v
            CK2= tf.Variable(tf.random.truncated_normal(shape=[hid_nodes_size, states_size], stddev=0.0001))
            DK3= tf.Variable(tf.random.truncated_normal(shape=[hid_nodes_size, input_size], stddev=0.0001))

            if is_bias:
                bxi = tf.Variable(tf.zeros(shape=[states_size]))
                bu = tf.Variable(tf.zeros(shape=[output_size]))
                bv = tf.Variable(tf.zeros(shape=[hid_nodes_size]))

            # feedford func for 1 step
            def feedforward(y, xi):
                #  v(k)    = CK2 xi(k) + DK3 y(k)
                v = tf.linalg.matmul(xi, CK2, transpose_b=True) + \
                    tf.linalg.matmul(y,  DK3, transpose_b=True)
                if is_bias:
                    v += bv

                #  z(k)    = phi(v(k))
                z = new_activation(v)

                #  w(k)    = (B-A/2) z(k)
                if np.size(Aphi) > 1:
                    w = tf.matmul(z, (Bphi - Aphi) / 2, transpose_b=True)
                else:
                    w = z * (Bphi - Aphi) / 2

                #  u(k)    = CK1 xi(k) + DK1 (B-A/2) z(k) + DK2 y(k)
                u = tf.linalg.matmul(xi, CK1, transpose_b=True) + \
                    tf.linalg.matmul(w,  DK1, transpose_b=True) + \
                    tf.linalg.matmul(y,  DK2, transpose_b=True)
                if is_bias:
                    u += bu

                #  xi(k+1) = AK  xi(k) + BK1 (B-A/2) z(k) + BK2 y(k)
                xi = tf.linalg.matmul(xi, AK,  transpose_b=True) + \
                     tf.linalg.matmul(w,  BK1, transpose_b=True) + \
                     tf.linalg.matmul(y,  BK2, transpose_b=True)
                if is_bias:
                    xi += bxi

                return u, xi

            # rnn feedforward
            xi = sy_initstate # may have to make shape consistent.
            outputs = []
            for i in range(len(sy_input)): # k steps
                y = sy_input[i]
                d = sy_done[i]

                u, xi = feedforward(y, xi)

                xi= xi*(1-d)+sy_initstate*d # if done

                outputs.append(u)

            output_ph = tf.stack(outputs, axis=1)

            # single step case for evaluation
            self.ph1_initstate_ns = tf.placeholder(shape=[None, states_size], name="xi1", dtype=tf.float32)
            self.ph1_input_no = tf.placeholder(shape=[None, input_size], name="ob1", dtype=tf.float32)

            # single step case for evaluation
            xi= self.ph1_initstate_ns
            y = self.ph1_input_no
            u, xi = feedforward(y, xi)

            self.sy1_output_na = u
            self.sy1_nextstate_ns = xi

            # save the attributes
            self.ob_dim = input_size
            self.ac_dim = output_size
            self.xi_dim = states_size
            self.hid_dim = hid_nodes_size

            self.activation = activation
            self.is_bias = is_bias

            self.ph_input_nto = input_placeholder
            self.ph_initstate_s = initstate_placeholder
            self.ph_ac_nta = tf.placeholder(shape=[None, step_num, output_size], name="acrnn", dtype=tf.float32)
            self.ph_adv_nt = tf.placeholder(shape=[None, step_num], name="advrnn", dtype=tf.float32)
            self.ph_dn_nt = termination_placeholder
            self.sy_output_nta = output_ph

            # phi
            self.Aphi = Aphi
            self.Bphi = Bphi

            # weights
            self.AK = AK
            self.BK1= BK1
            self.BK2= BK2
            self.CK1= CK1
            self.DK1= DK1
            self.DK2= DK2
            self.CK2= CK2
            self.DK3= DK3
            self.weights_assignment_buildup()

            if is_bias:
                self.bxi= bxi
                self.bu = bu
                self.bv = bv


            return output_ph

    def weights_assignment_buildup(self):

        self.aph_AK = tf.placeholder(shape=self.AK .get_shape().as_list(), name="aph_AK", dtype=tf.float32)
        self.aph_BK1= tf.placeholder(shape=self.BK1.get_shape().as_list(), name="aph_BK1", dtype=tf.float32)
        self.aph_BK2= tf.placeholder(shape=self.BK2.get_shape().as_list(), name="aph_BK2", dtype=tf.float32)
        self.aph_CK1= tf.placeholder(shape=self.CK1.get_shape().as_list(), name="aph_CK1", dtype=tf.float32)
        self.aph_DK1= tf.placeholder(shape=self.DK1.get_shape().as_list(), name="aph_DK1", dtype=tf.float32)
        self.aph_DK2= tf.placeholder(shape=self.DK2.get_shape().as_list(), name="aph_DK2", dtype=tf.float32)
        self.aph_CK2= tf.placeholder(shape=self.CK2.get_shape().as_list(), name="aph_CK2", dtype=tf.float32)
        self.aph_DK3= tf.placeholder(shape=self.DK3.get_shape().as_list(), name="aph_DK3", dtype=tf.float32)

        self.assign_AK = self.AK .assign(self.aph_AK )
        self.assign_BK1= self.BK1.assign(self.aph_BK1)
        self.assign_BK2= self.BK2.assign(self.aph_BK2)
        self.assign_CK1= self.CK1.assign(self.aph_CK1)
        self.assign_DK1= self.DK1.assign(self.aph_DK1)
        self.assign_DK2= self.DK2.assign(self.aph_DK2)
        self.assign_CK2= self.CK2.assign(self.aph_CK2)
        self.assign_DK3= self.DK3.assign(self.aph_DK3)

    def get_weights(self, sess, nobias=False):
        ret_var = [
            self.AK ,
            self.BK1,
            self.BK2,
            self.CK1,
            self.DK1,
            self.DK2,
            self.CK2,
            self.DK3
        ]
        ret_bias = [
            self.bxi,
            self.bu ,
            self.bv
        ]
        op = ret_var + ret_bias if self.is_bias else ret_var
        ret = sess.run(op)
        if not self.is_bias:
            ret += [None, None, None]
        return ret[:-3] if nobias else ret

    def set_weights(self, sess, AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3):
        sess.run([
            self.assign_AK ,
            self.assign_BK1,
            self.assign_BK2,
            self.assign_CK1,
            self.assign_DK1,
            self.assign_DK2,
            self.assign_CK2,
            self.assign_DK3
        ], feed_dict={
            self.aph_AK : AK ,
            self.aph_BK1: BK1,
            self.aph_BK2: BK2,
            self.aph_CK1: CK1,
            self.aph_DK1: DK1,
            self.aph_DK2: DK2,
            self.aph_CK2: CK2,
            self.aph_DK3: DK3
        })

    @staticmethod
    def build_tilde_activation(activation, Aphi=0, Bphi=1):
        """
        You do not need to call this by your self. Feed Aphi and Bphi to build will be ok.
        """

        def new_activation(v):
            w = activation(v)
            if np.size(Aphi) > 1:
                z = tf.matmul(w - tf.matmul(v, (Aphi + Bphi)/2, transpose_b=True),
                              2 * np.linalg.inv(Bphi - Aphi),   transpose_b=True)
            else:
                z = 2/(Bphi - Aphi) * (w - (Aphi + Bphi)/2 * v)
            return z

        return new_activation

