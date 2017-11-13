import tensorflow as tf

x = tf.constant([1,2,3,4,5,6,7,8,9], dtype=tf.float32)
y = tf.constant([10,100,1000,10000], dtype=tf.float32)
m = tf.constant([-20], dtype=tf.float32)

x_ = tf.expand_dims(x, 0)
y_ = tf.expand_dims(y, 1)
m_ = m

z = tf.add(tf.reshape(tf.add(x_, y_),[-1]),m_)

sess = tf.Session()

a=tf.Print(z,[z],message="This is a: ",summarize=100)
sess.run(a)
maset=set()
for i in range(25):
    print(i//5,i%5)

print("Len",len(maset))