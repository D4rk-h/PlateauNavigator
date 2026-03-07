OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[4];
h q[1];
cx q[2], q[3];
x q[1];
ccx q[0], q[1], q[2];
swap q[2], q[3];
id q[1];
y q[2];
rxx(pi/2) q[0], q[1];
measure q[0] -> c[0];