ߝ
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
?
	MLCConv2D

input"T
filter"T

unique_key"T*num_args
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)
"
	transposebool( "
num_argsint(
?
	MLCMatMul
a"T
b"T

unique_key"T*num_args
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2"
num_argsint ("

input_rankint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*	2.4.0-rc02v1.12.1-44683-gbcaa5ccc43e8??
?
conv1d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv1d_21/kernel
y
$conv1d_21/kernel/Read/ReadVariableOpReadVariableOpconv1d_21/kernel*"
_output_shapes
:@*
dtype0
t
conv1d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_21/bias
m
"conv1d_21/bias/Read/ReadVariableOpReadVariableOpconv1d_21/bias*
_output_shapes
:@*
dtype0
?
conv1d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_22/kernel
y
$conv1d_22/kernel/Read/ReadVariableOpReadVariableOpconv1d_22/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_22/bias
m
"conv1d_22/bias/Read/ReadVariableOpReadVariableOpconv1d_22/bias*
_output_shapes
:@*
dtype0
?
conv1d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_23/kernel
y
$conv1d_23/kernel/Read/ReadVariableOpReadVariableOpconv1d_23/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_23/bias
m
"conv1d_23/bias/Read/ReadVariableOpReadVariableOpconv1d_23/bias*
_output_shapes
:@*
dtype0
?
conv1d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_24/kernel
y
$conv1d_24/kernel/Read/ReadVariableOpReadVariableOpconv1d_24/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_24/bias
m
"conv1d_24/bias/Read/ReadVariableOpReadVariableOpconv1d_24/bias*
_output_shapes
:@*
dtype0
?
conv1d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_25/kernel
y
$conv1d_25/kernel/Read/ReadVariableOpReadVariableOpconv1d_25/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_25/bias
m
"conv1d_25/bias/Read/ReadVariableOpReadVariableOpconv1d_25/bias*
_output_shapes
:@*
dtype0
?
conv1d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_26/kernel
y
$conv1d_26/kernel/Read/ReadVariableOpReadVariableOpconv1d_26/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_26/bias
m
"conv1d_26/bias/Read/ReadVariableOpReadVariableOpconv1d_26/bias*
_output_shapes
:@*
dtype0
?
conv1d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_27/kernel
y
$conv1d_27/kernel/Read/ReadVariableOpReadVariableOpconv1d_27/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_27/bias
m
"conv1d_27/bias/Read/ReadVariableOpReadVariableOpconv1d_27/bias*
_output_shapes
:@*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
{
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_10/kernel
t
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes
:	?*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv1d_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv1d_21/kernel/m
?
+Adam/conv1d_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_21/kernel/m*"
_output_shapes
:@*
dtype0
?
Adam/conv1d_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_21/bias/m
{
)Adam/conv1d_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_21/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_22/kernel/m
?
+Adam/conv1d_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_22/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_22/bias/m
{
)Adam/conv1d_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_22/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_23/kernel/m
?
+Adam/conv1d_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_23/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_23/bias/m
{
)Adam/conv1d_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_23/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_24/kernel/m
?
+Adam/conv1d_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_24/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_24/bias/m
{
)Adam/conv1d_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_24/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_25/kernel/m
?
+Adam/conv1d_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_25/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_25/bias/m
{
)Adam/conv1d_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_25/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_26/kernel/m
?
+Adam/conv1d_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_26/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_26/bias/m
{
)Adam/conv1d_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_26/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_27/kernel/m
?
+Adam/conv1d_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_27/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_27/bias/m
{
)Adam/conv1d_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_27/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_10/kernel/m
?
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_11/kernel/m
?
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1d_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv1d_21/kernel/v
?
+Adam/conv1d_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_21/kernel/v*"
_output_shapes
:@*
dtype0
?
Adam/conv1d_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_21/bias/v
{
)Adam/conv1d_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_21/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_22/kernel/v
?
+Adam/conv1d_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_22/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_22/bias/v
{
)Adam/conv1d_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_22/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_23/kernel/v
?
+Adam/conv1d_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_23/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_23/bias/v
{
)Adam/conv1d_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_23/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_24/kernel/v
?
+Adam/conv1d_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_24/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_24/bias/v
{
)Adam/conv1d_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_24/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_25/kernel/v
?
+Adam/conv1d_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_25/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_25/bias/v
{
)Adam/conv1d_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_25/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_26/kernel/v
?
+Adam/conv1d_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_26/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_26/bias/v
{
)Adam/conv1d_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_26/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_27/kernel/v
?
+Adam/conv1d_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_27/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_27/bias/v
{
)Adam/conv1d_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_27/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_10/kernel/v
?
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_11/kernel/v
?
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?h
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?h
value?hB?h B?h
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
h

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
h

3kernel
4bias
5trainable_variables
6	variables
7regularization_losses
8	keras_api
 
h

9kernel
:bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
h

?kernel
@bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
R
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
R
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
h

Mkernel
Nbias
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
h

Skernel
Tbias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
?
Yiter

Zbeta_1

[beta_2
	\decay
]learning_ratem?m?m?m?!m?"m?'m?(m?-m?.m?3m?4m?9m?:m??m?@m?Mm?Nm?Sm?Tm?v?v?v?v?!v?"v?'v?(v?-v?.v?3v?4v?9v?:v??v?@v?Mv?Nv?Sv?Tv?
?
0
1
2
3
!4
"5
'6
(7
-8
.9
310
411
912
:13
?14
@15
M16
N17
S18
T19
?
0
1
2
3
!4
"5
'6
(7
-8
.9
310
411
912
:13
?14
@15
M16
N17
S18
T19
 
?
^layer_regularization_losses
_layer_metrics
trainable_variables

`layers
ametrics
	variables
regularization_losses
bnon_trainable_variables
 
\Z
VARIABLE_VALUEconv1d_21/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_21/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
clayer_regularization_losses
dlayer_metrics
trainable_variables

elayers
fmetrics
	variables
regularization_losses
gnon_trainable_variables
\Z
VARIABLE_VALUEconv1d_22/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_22/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
hlayer_regularization_losses
ilayer_metrics
trainable_variables

jlayers
kmetrics
	variables
regularization_losses
lnon_trainable_variables
\Z
VARIABLE_VALUEconv1d_23/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_23/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
?
mlayer_regularization_losses
nlayer_metrics
#trainable_variables

olayers
pmetrics
$	variables
%regularization_losses
qnon_trainable_variables
\Z
VARIABLE_VALUEconv1d_24/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_24/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
rlayer_regularization_losses
slayer_metrics
)trainable_variables

tlayers
umetrics
*	variables
+regularization_losses
vnon_trainable_variables
\Z
VARIABLE_VALUEconv1d_25/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_25/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
?
wlayer_regularization_losses
xlayer_metrics
/trainable_variables

ylayers
zmetrics
0	variables
1regularization_losses
{non_trainable_variables
\Z
VARIABLE_VALUEconv1d_26/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_26/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 
?
|layer_regularization_losses
}layer_metrics
5trainable_variables

~layers
metrics
6	variables
7regularization_losses
?non_trainable_variables
\Z
VARIABLE_VALUEconv1d_27/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_27/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1

90
:1
 
?
 ?layer_regularization_losses
?layer_metrics
;trainable_variables
?layers
?metrics
<	variables
=regularization_losses
?non_trainable_variables
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
?
 ?layer_regularization_losses
?layer_metrics
Atrainable_variables
?layers
?metrics
B	variables
Cregularization_losses
?non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
Etrainable_variables
?layers
?metrics
F	variables
Gregularization_losses
?non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
Itrainable_variables
?layers
?metrics
J	variables
Kregularization_losses
?non_trainable_variables
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

M0
N1

M0
N1
 
?
 ?layer_regularization_losses
?layer_metrics
Otrainable_variables
?layers
?metrics
P	variables
Qregularization_losses
?non_trainable_variables
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

S0
T1
 
?
 ?layer_regularization_losses
?layer_metrics
Utrainable_variables
?layers
?metrics
V	variables
Wregularization_losses
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv1d_21/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_21/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_22/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_22/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_23/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_23/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_24/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_24/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_25/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_25/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_26/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_26/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_27/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_27/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_21/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_21/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_22/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_22/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_23/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_23/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_24/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_24/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_25/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_25/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_26/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_26/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_27/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_27/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
v
serving_default_catPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_convPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_catserving_default_convconv1d_21/kernelconv1d_21/biasconv1d_22/kernelconv1d_22/biasconv1d_23/kernelconv1d_23/biasconv1d_24/kernelconv1d_24/biasconv1d_25/kernelconv1d_25/biasconv1d_26/kernelconv1d_26/biasconv1d_27/kernelconv1d_27/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_231067
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv1d_21/kernel/Read/ReadVariableOp"conv1d_21/bias/Read/ReadVariableOp$conv1d_22/kernel/Read/ReadVariableOp"conv1d_22/bias/Read/ReadVariableOp$conv1d_23/kernel/Read/ReadVariableOp"conv1d_23/bias/Read/ReadVariableOp$conv1d_24/kernel/Read/ReadVariableOp"conv1d_24/bias/Read/ReadVariableOp$conv1d_25/kernel/Read/ReadVariableOp"conv1d_25/bias/Read/ReadVariableOp$conv1d_26/kernel/Read/ReadVariableOp"conv1d_26/bias/Read/ReadVariableOp$conv1d_27/kernel/Read/ReadVariableOp"conv1d_27/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv1d_21/kernel/m/Read/ReadVariableOp)Adam/conv1d_21/bias/m/Read/ReadVariableOp+Adam/conv1d_22/kernel/m/Read/ReadVariableOp)Adam/conv1d_22/bias/m/Read/ReadVariableOp+Adam/conv1d_23/kernel/m/Read/ReadVariableOp)Adam/conv1d_23/bias/m/Read/ReadVariableOp+Adam/conv1d_24/kernel/m/Read/ReadVariableOp)Adam/conv1d_24/bias/m/Read/ReadVariableOp+Adam/conv1d_25/kernel/m/Read/ReadVariableOp)Adam/conv1d_25/bias/m/Read/ReadVariableOp+Adam/conv1d_26/kernel/m/Read/ReadVariableOp)Adam/conv1d_26/bias/m/Read/ReadVariableOp+Adam/conv1d_27/kernel/m/Read/ReadVariableOp)Adam/conv1d_27/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp+Adam/conv1d_21/kernel/v/Read/ReadVariableOp)Adam/conv1d_21/bias/v/Read/ReadVariableOp+Adam/conv1d_22/kernel/v/Read/ReadVariableOp)Adam/conv1d_22/bias/v/Read/ReadVariableOp+Adam/conv1d_23/kernel/v/Read/ReadVariableOp)Adam/conv1d_23/bias/v/Read/ReadVariableOp+Adam/conv1d_24/kernel/v/Read/ReadVariableOp)Adam/conv1d_24/bias/v/Read/ReadVariableOp+Adam/conv1d_25/kernel/v/Read/ReadVariableOp)Adam/conv1d_25/bias/v/Read/ReadVariableOp+Adam/conv1d_26/kernel/v/Read/ReadVariableOp)Adam/conv1d_26/bias/v/Read/ReadVariableOp+Adam/conv1d_27/kernel/v/Read/ReadVariableOp)Adam/conv1d_27/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOpConst*R
TinK
I2G	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_231919
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_21/kernelconv1d_21/biasconv1d_22/kernelconv1d_22/biasconv1d_23/kernelconv1d_23/biasconv1d_24/kernelconv1d_24/biasconv1d_25/kernelconv1d_25/biasconv1d_26/kernelconv1d_26/biasconv1d_27/kernelconv1d_27/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv1d_21/kernel/mAdam/conv1d_21/bias/mAdam/conv1d_22/kernel/mAdam/conv1d_22/bias/mAdam/conv1d_23/kernel/mAdam/conv1d_23/bias/mAdam/conv1d_24/kernel/mAdam/conv1d_24/bias/mAdam/conv1d_25/kernel/mAdam/conv1d_25/bias/mAdam/conv1d_26/kernel/mAdam/conv1d_26/bias/mAdam/conv1d_27/kernel/mAdam/conv1d_27/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/dense_10/kernel/mAdam/dense_10/bias/mAdam/dense_11/kernel/mAdam/dense_11/bias/mAdam/conv1d_21/kernel/vAdam/conv1d_21/bias/vAdam/conv1d_22/kernel/vAdam/conv1d_22/bias/vAdam/conv1d_23/kernel/vAdam/conv1d_23/bias/vAdam/conv1d_24/kernel/vAdam/conv1d_24/bias/vAdam/conv1d_25/kernel/vAdam/conv1d_25/bias/vAdam/conv1d_26/kernel/vAdam/conv1d_26/bias/vAdam/conv1d_27/kernel/vAdam/conv1d_27/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/vAdam/dense_10/kernel/vAdam/dense_10/bias/vAdam/dense_11/kernel/vAdam/dense_11/bias/v*Q
TinJ
H2F*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_232136??
?
?
E__inference_conv1d_23_layer_call_and_return_conditional_losses_231487

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv1d_25_layer_call_and_return_conditional_losses_231541

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?;
?
C__inference_model_3_layer_call_and_return_conditional_losses_230804
conv
cat
conv1d_21_230751
conv1d_21_230753
conv1d_22_230756
conv1d_22_230758
conv1d_23_230761
conv1d_23_230763
conv1d_24_230766
conv1d_24_230768
conv1d_25_230771
conv1d_25_230773
conv1d_26_230776
conv1d_26_230778
conv1d_27_230781
conv1d_27_230783
dense_9_230786
dense_9_230788
dense_10_230793
dense_10_230795
dense_11_230798
dense_11_230800
identity??!conv1d_21/StatefulPartitionedCall?!conv1d_22/StatefulPartitionedCall?!conv1d_23/StatefulPartitionedCall?!conv1d_24/StatefulPartitionedCall?!conv1d_25/StatefulPartitionedCall?!conv1d_26/StatefulPartitionedCall?!conv1d_27/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
!conv1d_21/StatefulPartitionedCallStatefulPartitionedCallconvconv1d_21_230751conv1d_21_230753*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_21_layer_call_and_return_conditional_losses_2304152#
!conv1d_21/StatefulPartitionedCall?
!conv1d_22/StatefulPartitionedCallStatefulPartitionedCall*conv1d_21/StatefulPartitionedCall:output:0conv1d_22_230756conv1d_22_230758*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_22_layer_call_and_return_conditional_losses_2304492#
!conv1d_22/StatefulPartitionedCall?
!conv1d_23/StatefulPartitionedCallStatefulPartitionedCall*conv1d_22/StatefulPartitionedCall:output:0conv1d_23_230761conv1d_23_230763*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_23_layer_call_and_return_conditional_losses_2304832#
!conv1d_23/StatefulPartitionedCall?
!conv1d_24/StatefulPartitionedCallStatefulPartitionedCall*conv1d_23/StatefulPartitionedCall:output:0conv1d_24_230766conv1d_24_230768*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_24_layer_call_and_return_conditional_losses_2305172#
!conv1d_24/StatefulPartitionedCall?
!conv1d_25/StatefulPartitionedCallStatefulPartitionedCall*conv1d_24/StatefulPartitionedCall:output:0conv1d_25_230771conv1d_25_230773*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_25_layer_call_and_return_conditional_losses_2305512#
!conv1d_25/StatefulPartitionedCall?
!conv1d_26/StatefulPartitionedCallStatefulPartitionedCall*conv1d_25/StatefulPartitionedCall:output:0conv1d_26_230776conv1d_26_230778*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_26_layer_call_and_return_conditional_losses_2305852#
!conv1d_26/StatefulPartitionedCall?
!conv1d_27/StatefulPartitionedCallStatefulPartitionedCall*conv1d_26/StatefulPartitionedCall:output:0conv1d_27_230781conv1d_27_230783*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_27_layer_call_and_return_conditional_losses_2306192#
!conv1d_27/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallcatdense_9_230786dense_9_230788*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2306462!
dense_9/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall*conv1d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2306682
flatten_3/PartitionedCall?
concatenate_3/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_2306832
concatenate_3/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_10_230793dense_10_230795*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_2307032"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_230798dense_11_230800*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_2307302"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0"^conv1d_21/StatefulPartitionedCall"^conv1d_22/StatefulPartitionedCall"^conv1d_23/StatefulPartitionedCall"^conv1d_24/StatefulPartitionedCall"^conv1d_25/StatefulPartitionedCall"^conv1d_26/StatefulPartitionedCall"^conv1d_27/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2F
!conv1d_21/StatefulPartitionedCall!conv1d_21/StatefulPartitionedCall2F
!conv1d_22/StatefulPartitionedCall!conv1d_22/StatefulPartitionedCall2F
!conv1d_23/StatefulPartitionedCall!conv1d_23/StatefulPartitionedCall2F
!conv1d_24/StatefulPartitionedCall!conv1d_24/StatefulPartitionedCall2F
!conv1d_25/StatefulPartitionedCall!conv1d_25/StatefulPartitionedCall2F
!conv1d_26/StatefulPartitionedCall!conv1d_26/StatefulPartitionedCall2F
!conv1d_27/StatefulPartitionedCall!conv1d_27/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
?
?
E__inference_conv1d_22_layer_call_and_return_conditional_losses_231460

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
~
)__inference_dense_11_layer_call_fn_231688

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_2307302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_3_layer_call_fn_231635

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2306682
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?#
"__inference__traced_restore_232136
file_prefix%
!assignvariableop_conv1d_21_kernel%
!assignvariableop_1_conv1d_21_bias'
#assignvariableop_2_conv1d_22_kernel%
!assignvariableop_3_conv1d_22_bias'
#assignvariableop_4_conv1d_23_kernel%
!assignvariableop_5_conv1d_23_bias'
#assignvariableop_6_conv1d_24_kernel%
!assignvariableop_7_conv1d_24_bias'
#assignvariableop_8_conv1d_25_kernel%
!assignvariableop_9_conv1d_25_bias(
$assignvariableop_10_conv1d_26_kernel&
"assignvariableop_11_conv1d_26_bias(
$assignvariableop_12_conv1d_27_kernel&
"assignvariableop_13_conv1d_27_bias&
"assignvariableop_14_dense_9_kernel$
 assignvariableop_15_dense_9_bias'
#assignvariableop_16_dense_10_kernel%
!assignvariableop_17_dense_10_bias'
#assignvariableop_18_dense_11_kernel%
!assignvariableop_19_dense_11_bias!
assignvariableop_20_adam_iter#
assignvariableop_21_adam_beta_1#
assignvariableop_22_adam_beta_2"
assignvariableop_23_adam_decay*
&assignvariableop_24_adam_learning_rate
assignvariableop_25_total
assignvariableop_26_count
assignvariableop_27_total_1
assignvariableop_28_count_1/
+assignvariableop_29_adam_conv1d_21_kernel_m-
)assignvariableop_30_adam_conv1d_21_bias_m/
+assignvariableop_31_adam_conv1d_22_kernel_m-
)assignvariableop_32_adam_conv1d_22_bias_m/
+assignvariableop_33_adam_conv1d_23_kernel_m-
)assignvariableop_34_adam_conv1d_23_bias_m/
+assignvariableop_35_adam_conv1d_24_kernel_m-
)assignvariableop_36_adam_conv1d_24_bias_m/
+assignvariableop_37_adam_conv1d_25_kernel_m-
)assignvariableop_38_adam_conv1d_25_bias_m/
+assignvariableop_39_adam_conv1d_26_kernel_m-
)assignvariableop_40_adam_conv1d_26_bias_m/
+assignvariableop_41_adam_conv1d_27_kernel_m-
)assignvariableop_42_adam_conv1d_27_bias_m-
)assignvariableop_43_adam_dense_9_kernel_m+
'assignvariableop_44_adam_dense_9_bias_m.
*assignvariableop_45_adam_dense_10_kernel_m,
(assignvariableop_46_adam_dense_10_bias_m.
*assignvariableop_47_adam_dense_11_kernel_m,
(assignvariableop_48_adam_dense_11_bias_m/
+assignvariableop_49_adam_conv1d_21_kernel_v-
)assignvariableop_50_adam_conv1d_21_bias_v/
+assignvariableop_51_adam_conv1d_22_kernel_v-
)assignvariableop_52_adam_conv1d_22_bias_v/
+assignvariableop_53_adam_conv1d_23_kernel_v-
)assignvariableop_54_adam_conv1d_23_bias_v/
+assignvariableop_55_adam_conv1d_24_kernel_v-
)assignvariableop_56_adam_conv1d_24_bias_v/
+assignvariableop_57_adam_conv1d_25_kernel_v-
)assignvariableop_58_adam_conv1d_25_bias_v/
+assignvariableop_59_adam_conv1d_26_kernel_v-
)assignvariableop_60_adam_conv1d_26_bias_v/
+assignvariableop_61_adam_conv1d_27_kernel_v-
)assignvariableop_62_adam_conv1d_27_bias_v-
)assignvariableop_63_adam_dense_9_kernel_v+
'assignvariableop_64_adam_dense_9_bias_v.
*assignvariableop_65_adam_dense_10_kernel_v,
(assignvariableop_66_adam_dense_10_bias_v.
*assignvariableop_67_adam_dense_11_kernel_v,
(assignvariableop_68_adam_dense_11_bias_v
identity_70??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?&
value?&B?&FB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*T
dtypesJ
H2F	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_21_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_21_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv1d_22_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv1d_22_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv1d_23_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv1d_23_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv1d_24_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv1d_24_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv1d_25_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv1d_25_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv1d_26_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv1d_26_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv1d_27_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv1d_27_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_9_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_9_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_10_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_10_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_11_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_11_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv1d_21_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv1d_21_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv1d_22_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv1d_22_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv1d_23_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv1d_23_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv1d_24_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv1d_24_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv1d_25_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv1d_25_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv1d_26_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv1d_26_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv1d_27_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv1d_27_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_9_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_9_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_10_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_10_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_11_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_11_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv1d_21_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv1d_21_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv1d_22_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv1d_22_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv1d_23_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv1d_23_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv1d_24_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv1d_24_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv1d_25_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv1d_25_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv1d_26_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv1d_26_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv1d_27_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv1d_27_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adam_dense_9_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp'assignvariableop_64_adam_dense_9_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_10_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_10_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_11_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_11_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_689
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_69Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_69?
Identity_70IdentityIdentity_69:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_70"#
identity_70Identity_70:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
(__inference_model_3_layer_call_fn_231011
conv
cat
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconvcatunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_2309682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
?
?
E__inference_conv1d_23_layer_call_and_return_conditional_losses_230483

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
u
I__inference_concatenate_3_layer_call_and_return_conditional_losses_231642
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:??????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
(__inference_model_3_layer_call_fn_230908
conv
cat
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconvcatunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_2308652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
?

*__inference_conv1d_22_layer_call_fn_231469

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_22_layer_call_and_return_conditional_losses_2304492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

*__inference_conv1d_27_layer_call_fn_231604

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_27_layer_call_and_return_conditional_losses_2306192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv1d_22_layer_call_and_return_conditional_losses_230449

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv1d_21_layer_call_and_return_conditional_losses_230415

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_dense_9_layer_call_and_return_conditional_losses_231615

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_conv1d_23_layer_call_fn_231496

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_23_layer_call_and_return_conditional_losses_2304832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv1d_26_layer_call_and_return_conditional_losses_231568

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
Z
.__inference_concatenate_3_layer_call_fn_231648
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_2306832
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:??????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
~
)__inference_dense_10_layer_call_fn_231668

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_2307032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv1d_27_layer_call_and_return_conditional_losses_230619

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

*__inference_conv1d_26_layer_call_fn_231577

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_26_layer_call_and_return_conditional_losses_2305852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
}
(__inference_dense_9_layer_call_fn_231624

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2306462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_conv1d_24_layer_call_fn_231523

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_24_layer_call_and_return_conditional_losses_2305172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?;
?
C__inference_model_3_layer_call_and_return_conditional_losses_230747
conv
cat
conv1d_21_230426
conv1d_21_230428
conv1d_22_230460
conv1d_22_230462
conv1d_23_230494
conv1d_23_230496
conv1d_24_230528
conv1d_24_230530
conv1d_25_230562
conv1d_25_230564
conv1d_26_230596
conv1d_26_230598
conv1d_27_230630
conv1d_27_230632
dense_9_230657
dense_9_230659
dense_10_230714
dense_10_230716
dense_11_230741
dense_11_230743
identity??!conv1d_21/StatefulPartitionedCall?!conv1d_22/StatefulPartitionedCall?!conv1d_23/StatefulPartitionedCall?!conv1d_24/StatefulPartitionedCall?!conv1d_25/StatefulPartitionedCall?!conv1d_26/StatefulPartitionedCall?!conv1d_27/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
!conv1d_21/StatefulPartitionedCallStatefulPartitionedCallconvconv1d_21_230426conv1d_21_230428*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_21_layer_call_and_return_conditional_losses_2304152#
!conv1d_21/StatefulPartitionedCall?
!conv1d_22/StatefulPartitionedCallStatefulPartitionedCall*conv1d_21/StatefulPartitionedCall:output:0conv1d_22_230460conv1d_22_230462*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_22_layer_call_and_return_conditional_losses_2304492#
!conv1d_22/StatefulPartitionedCall?
!conv1d_23/StatefulPartitionedCallStatefulPartitionedCall*conv1d_22/StatefulPartitionedCall:output:0conv1d_23_230494conv1d_23_230496*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_23_layer_call_and_return_conditional_losses_2304832#
!conv1d_23/StatefulPartitionedCall?
!conv1d_24/StatefulPartitionedCallStatefulPartitionedCall*conv1d_23/StatefulPartitionedCall:output:0conv1d_24_230528conv1d_24_230530*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_24_layer_call_and_return_conditional_losses_2305172#
!conv1d_24/StatefulPartitionedCall?
!conv1d_25/StatefulPartitionedCallStatefulPartitionedCall*conv1d_24/StatefulPartitionedCall:output:0conv1d_25_230562conv1d_25_230564*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_25_layer_call_and_return_conditional_losses_2305512#
!conv1d_25/StatefulPartitionedCall?
!conv1d_26/StatefulPartitionedCallStatefulPartitionedCall*conv1d_25/StatefulPartitionedCall:output:0conv1d_26_230596conv1d_26_230598*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_26_layer_call_and_return_conditional_losses_2305852#
!conv1d_26/StatefulPartitionedCall?
!conv1d_27/StatefulPartitionedCallStatefulPartitionedCall*conv1d_26/StatefulPartitionedCall:output:0conv1d_27_230630conv1d_27_230632*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_27_layer_call_and_return_conditional_losses_2306192#
!conv1d_27/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallcatdense_9_230657dense_9_230659*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2306462!
dense_9/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall*conv1d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2306682
flatten_3/PartitionedCall?
concatenate_3/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_2306832
concatenate_3/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_10_230714dense_10_230716*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_2307032"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_230741dense_11_230743*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_2307302"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0"^conv1d_21/StatefulPartitionedCall"^conv1d_22/StatefulPartitionedCall"^conv1d_23/StatefulPartitionedCall"^conv1d_24/StatefulPartitionedCall"^conv1d_25/StatefulPartitionedCall"^conv1d_26/StatefulPartitionedCall"^conv1d_27/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2F
!conv1d_21/StatefulPartitionedCall!conv1d_21/StatefulPartitionedCall2F
!conv1d_22/StatefulPartitionedCall!conv1d_22/StatefulPartitionedCall2F
!conv1d_23/StatefulPartitionedCall!conv1d_23/StatefulPartitionedCall2F
!conv1d_24/StatefulPartitionedCall!conv1d_24/StatefulPartitionedCall2F
!conv1d_25/StatefulPartitionedCall!conv1d_25/StatefulPartitionedCall2F
!conv1d_26/StatefulPartitionedCall!conv1d_26/StatefulPartitionedCall2F
!conv1d_27/StatefulPartitionedCall!conv1d_27/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
?

?
D__inference_dense_11_layer_call_and_return_conditional_losses_231679

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_231067
cat
conv
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconvcatunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_2303922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:?????????

_user_specified_namecat:QM
+
_output_shapes
:?????????

_user_specified_nameconv
?
?
E__inference_conv1d_26_layer_call_and_return_conditional_losses_230585

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv1d_25_layer_call_and_return_conditional_losses_230551

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_dense_10_layer_call_and_return_conditional_losses_231659

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_model_3_layer_call_fn_231415
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_2309682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
C__inference_dense_9_layer_call_and_return_conditional_losses_230646

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_conv1d_25_layer_call_fn_231550

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_25_layer_call_and_return_conditional_losses_2305512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv1d_21_layer_call_and_return_conditional_losses_231433

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?;
?
C__inference_model_3_layer_call_and_return_conditional_losses_230968

inputs
inputs_1
conv1d_21_230915
conv1d_21_230917
conv1d_22_230920
conv1d_22_230922
conv1d_23_230925
conv1d_23_230927
conv1d_24_230930
conv1d_24_230932
conv1d_25_230935
conv1d_25_230937
conv1d_26_230940
conv1d_26_230942
conv1d_27_230945
conv1d_27_230947
dense_9_230950
dense_9_230952
dense_10_230957
dense_10_230959
dense_11_230962
dense_11_230964
identity??!conv1d_21/StatefulPartitionedCall?!conv1d_22/StatefulPartitionedCall?!conv1d_23/StatefulPartitionedCall?!conv1d_24/StatefulPartitionedCall?!conv1d_25/StatefulPartitionedCall?!conv1d_26/StatefulPartitionedCall?!conv1d_27/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
!conv1d_21/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_21_230915conv1d_21_230917*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_21_layer_call_and_return_conditional_losses_2304152#
!conv1d_21/StatefulPartitionedCall?
!conv1d_22/StatefulPartitionedCallStatefulPartitionedCall*conv1d_21/StatefulPartitionedCall:output:0conv1d_22_230920conv1d_22_230922*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_22_layer_call_and_return_conditional_losses_2304492#
!conv1d_22/StatefulPartitionedCall?
!conv1d_23/StatefulPartitionedCallStatefulPartitionedCall*conv1d_22/StatefulPartitionedCall:output:0conv1d_23_230925conv1d_23_230927*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_23_layer_call_and_return_conditional_losses_2304832#
!conv1d_23/StatefulPartitionedCall?
!conv1d_24/StatefulPartitionedCallStatefulPartitionedCall*conv1d_23/StatefulPartitionedCall:output:0conv1d_24_230930conv1d_24_230932*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_24_layer_call_and_return_conditional_losses_2305172#
!conv1d_24/StatefulPartitionedCall?
!conv1d_25/StatefulPartitionedCallStatefulPartitionedCall*conv1d_24/StatefulPartitionedCall:output:0conv1d_25_230935conv1d_25_230937*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_25_layer_call_and_return_conditional_losses_2305512#
!conv1d_25/StatefulPartitionedCall?
!conv1d_26/StatefulPartitionedCallStatefulPartitionedCall*conv1d_25/StatefulPartitionedCall:output:0conv1d_26_230940conv1d_26_230942*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_26_layer_call_and_return_conditional_losses_2305852#
!conv1d_26/StatefulPartitionedCall?
!conv1d_27/StatefulPartitionedCallStatefulPartitionedCall*conv1d_26/StatefulPartitionedCall:output:0conv1d_27_230945conv1d_27_230947*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_27_layer_call_and_return_conditional_losses_2306192#
!conv1d_27/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_9_230950dense_9_230952*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2306462!
dense_9/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall*conv1d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2306682
flatten_3/PartitionedCall?
concatenate_3/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_2306832
concatenate_3/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_10_230957dense_10_230959*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_2307032"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_230962dense_11_230964*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_2307302"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0"^conv1d_21/StatefulPartitionedCall"^conv1d_22/StatefulPartitionedCall"^conv1d_23/StatefulPartitionedCall"^conv1d_24/StatefulPartitionedCall"^conv1d_25/StatefulPartitionedCall"^conv1d_26/StatefulPartitionedCall"^conv1d_27/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2F
!conv1d_21/StatefulPartitionedCall!conv1d_21/StatefulPartitionedCall2F
!conv1d_22/StatefulPartitionedCall!conv1d_22/StatefulPartitionedCall2F
!conv1d_23/StatefulPartitionedCall!conv1d_23/StatefulPartitionedCall2F
!conv1d_24/StatefulPartitionedCall!conv1d_24/StatefulPartitionedCall2F
!conv1d_25/StatefulPartitionedCall!conv1d_25/StatefulPartitionedCall2F
!conv1d_26/StatefulPartitionedCall!conv1d_26/StatefulPartitionedCall2F
!conv1d_27/StatefulPartitionedCall!conv1d_27/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_dense_10_layer_call_and_return_conditional_losses_230703

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?;
?
C__inference_model_3_layer_call_and_return_conditional_losses_230865

inputs
inputs_1
conv1d_21_230812
conv1d_21_230814
conv1d_22_230817
conv1d_22_230819
conv1d_23_230822
conv1d_23_230824
conv1d_24_230827
conv1d_24_230829
conv1d_25_230832
conv1d_25_230834
conv1d_26_230837
conv1d_26_230839
conv1d_27_230842
conv1d_27_230844
dense_9_230847
dense_9_230849
dense_10_230854
dense_10_230856
dense_11_230859
dense_11_230861
identity??!conv1d_21/StatefulPartitionedCall?!conv1d_22/StatefulPartitionedCall?!conv1d_23/StatefulPartitionedCall?!conv1d_24/StatefulPartitionedCall?!conv1d_25/StatefulPartitionedCall?!conv1d_26/StatefulPartitionedCall?!conv1d_27/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
!conv1d_21/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_21_230812conv1d_21_230814*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_21_layer_call_and_return_conditional_losses_2304152#
!conv1d_21/StatefulPartitionedCall?
!conv1d_22/StatefulPartitionedCallStatefulPartitionedCall*conv1d_21/StatefulPartitionedCall:output:0conv1d_22_230817conv1d_22_230819*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_22_layer_call_and_return_conditional_losses_2304492#
!conv1d_22/StatefulPartitionedCall?
!conv1d_23/StatefulPartitionedCallStatefulPartitionedCall*conv1d_22/StatefulPartitionedCall:output:0conv1d_23_230822conv1d_23_230824*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_23_layer_call_and_return_conditional_losses_2304832#
!conv1d_23/StatefulPartitionedCall?
!conv1d_24/StatefulPartitionedCallStatefulPartitionedCall*conv1d_23/StatefulPartitionedCall:output:0conv1d_24_230827conv1d_24_230829*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_24_layer_call_and_return_conditional_losses_2305172#
!conv1d_24/StatefulPartitionedCall?
!conv1d_25/StatefulPartitionedCallStatefulPartitionedCall*conv1d_24/StatefulPartitionedCall:output:0conv1d_25_230832conv1d_25_230834*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_25_layer_call_and_return_conditional_losses_2305512#
!conv1d_25/StatefulPartitionedCall?
!conv1d_26/StatefulPartitionedCallStatefulPartitionedCall*conv1d_25/StatefulPartitionedCall:output:0conv1d_26_230837conv1d_26_230839*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_26_layer_call_and_return_conditional_losses_2305852#
!conv1d_26/StatefulPartitionedCall?
!conv1d_27/StatefulPartitionedCallStatefulPartitionedCall*conv1d_26/StatefulPartitionedCall:output:0conv1d_27_230842conv1d_27_230844*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_27_layer_call_and_return_conditional_losses_2306192#
!conv1d_27/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_9_230847dense_9_230849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2306462!
dense_9/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall*conv1d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2306682
flatten_3/PartitionedCall?
concatenate_3/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_2306832
concatenate_3/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_10_230854dense_10_230856*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_2307032"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_230859dense_11_230861*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_2307302"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0"^conv1d_21/StatefulPartitionedCall"^conv1d_22/StatefulPartitionedCall"^conv1d_23/StatefulPartitionedCall"^conv1d_24/StatefulPartitionedCall"^conv1d_25/StatefulPartitionedCall"^conv1d_26/StatefulPartitionedCall"^conv1d_27/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2F
!conv1d_21/StatefulPartitionedCall!conv1d_21/StatefulPartitionedCall2F
!conv1d_22/StatefulPartitionedCall!conv1d_22/StatefulPartitionedCall2F
!conv1d_23/StatefulPartitionedCall!conv1d_23/StatefulPartitionedCall2F
!conv1d_24/StatefulPartitionedCall!conv1d_24/StatefulPartitionedCall2F
!conv1d_25/StatefulPartitionedCall!conv1d_25/StatefulPartitionedCall2F
!conv1d_26/StatefulPartitionedCall!conv1d_26/StatefulPartitionedCall2F
!conv1d_27/StatefulPartitionedCall!conv1d_27/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_conv1d_21_layer_call_fn_231442

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_21_layer_call_and_return_conditional_losses_2304152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv1d_24_layer_call_and_return_conditional_losses_230517

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv1d_27_layer_call_and_return_conditional_losses_231595

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
s
I__inference_concatenate_3_layer_call_and_return_conditional_losses_230683

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:??????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
__inference__traced_save_231919
file_prefix/
+savev2_conv1d_21_kernel_read_readvariableop-
)savev2_conv1d_21_bias_read_readvariableop/
+savev2_conv1d_22_kernel_read_readvariableop-
)savev2_conv1d_22_bias_read_readvariableop/
+savev2_conv1d_23_kernel_read_readvariableop-
)savev2_conv1d_23_bias_read_readvariableop/
+savev2_conv1d_24_kernel_read_readvariableop-
)savev2_conv1d_24_bias_read_readvariableop/
+savev2_conv1d_25_kernel_read_readvariableop-
)savev2_conv1d_25_bias_read_readvariableop/
+savev2_conv1d_26_kernel_read_readvariableop-
)savev2_conv1d_26_bias_read_readvariableop/
+savev2_conv1d_27_kernel_read_readvariableop-
)savev2_conv1d_27_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv1d_21_kernel_m_read_readvariableop4
0savev2_adam_conv1d_21_bias_m_read_readvariableop6
2savev2_adam_conv1d_22_kernel_m_read_readvariableop4
0savev2_adam_conv1d_22_bias_m_read_readvariableop6
2savev2_adam_conv1d_23_kernel_m_read_readvariableop4
0savev2_adam_conv1d_23_bias_m_read_readvariableop6
2savev2_adam_conv1d_24_kernel_m_read_readvariableop4
0savev2_adam_conv1d_24_bias_m_read_readvariableop6
2savev2_adam_conv1d_25_kernel_m_read_readvariableop4
0savev2_adam_conv1d_25_bias_m_read_readvariableop6
2savev2_adam_conv1d_26_kernel_m_read_readvariableop4
0savev2_adam_conv1d_26_bias_m_read_readvariableop6
2savev2_adam_conv1d_27_kernel_m_read_readvariableop4
0savev2_adam_conv1d_27_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableop6
2savev2_adam_conv1d_21_kernel_v_read_readvariableop4
0savev2_adam_conv1d_21_bias_v_read_readvariableop6
2savev2_adam_conv1d_22_kernel_v_read_readvariableop4
0savev2_adam_conv1d_22_bias_v_read_readvariableop6
2savev2_adam_conv1d_23_kernel_v_read_readvariableop4
0savev2_adam_conv1d_23_bias_v_read_readvariableop6
2savev2_adam_conv1d_24_kernel_v_read_readvariableop4
0savev2_adam_conv1d_24_bias_v_read_readvariableop6
2savev2_adam_conv1d_25_kernel_v_read_readvariableop4
0savev2_adam_conv1d_25_bias_v_read_readvariableop6
2savev2_adam_conv1d_26_kernel_v_read_readvariableop4
0savev2_adam_conv1d_26_bias_v_read_readvariableop6
2savev2_adam_conv1d_27_kernel_v_read_readvariableop4
0savev2_adam_conv1d_27_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?&
value?&B?&FB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv1d_21_kernel_read_readvariableop)savev2_conv1d_21_bias_read_readvariableop+savev2_conv1d_22_kernel_read_readvariableop)savev2_conv1d_22_bias_read_readvariableop+savev2_conv1d_23_kernel_read_readvariableop)savev2_conv1d_23_bias_read_readvariableop+savev2_conv1d_24_kernel_read_readvariableop)savev2_conv1d_24_bias_read_readvariableop+savev2_conv1d_25_kernel_read_readvariableop)savev2_conv1d_25_bias_read_readvariableop+savev2_conv1d_26_kernel_read_readvariableop)savev2_conv1d_26_bias_read_readvariableop+savev2_conv1d_27_kernel_read_readvariableop)savev2_conv1d_27_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv1d_21_kernel_m_read_readvariableop0savev2_adam_conv1d_21_bias_m_read_readvariableop2savev2_adam_conv1d_22_kernel_m_read_readvariableop0savev2_adam_conv1d_22_bias_m_read_readvariableop2savev2_adam_conv1d_23_kernel_m_read_readvariableop0savev2_adam_conv1d_23_bias_m_read_readvariableop2savev2_adam_conv1d_24_kernel_m_read_readvariableop0savev2_adam_conv1d_24_bias_m_read_readvariableop2savev2_adam_conv1d_25_kernel_m_read_readvariableop0savev2_adam_conv1d_25_bias_m_read_readvariableop2savev2_adam_conv1d_26_kernel_m_read_readvariableop0savev2_adam_conv1d_26_bias_m_read_readvariableop2savev2_adam_conv1d_27_kernel_m_read_readvariableop0savev2_adam_conv1d_27_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableop2savev2_adam_conv1d_21_kernel_v_read_readvariableop0savev2_adam_conv1d_21_bias_v_read_readvariableop2savev2_adam_conv1d_22_kernel_v_read_readvariableop0savev2_adam_conv1d_22_bias_v_read_readvariableop2savev2_adam_conv1d_23_kernel_v_read_readvariableop0savev2_adam_conv1d_23_bias_v_read_readvariableop2savev2_adam_conv1d_24_kernel_v_read_readvariableop0savev2_adam_conv1d_24_bias_v_read_readvariableop2savev2_adam_conv1d_25_kernel_v_read_readvariableop0savev2_adam_conv1d_25_bias_v_read_readvariableop2savev2_adam_conv1d_26_kernel_v_read_readvariableop0savev2_adam_conv1d_26_bias_v_read_readvariableop2savev2_adam_conv1d_27_kernel_v_read_readvariableop0savev2_adam_conv1d_27_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *T
dtypesJ
H2F	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:::	?:::: : : : : : : : : :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:::	?::::@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:::	?:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:(	$
"
_output_shapes
:@@: 


_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:@: 

_output_shapes
:@:( $
"
_output_shapes
:@@: !

_output_shapes
:@:("$
"
_output_shapes
:@@: #

_output_shapes
:@:($$
"
_output_shapes
:@@: %

_output_shapes
:@:(&$
"
_output_shapes
:@@: '

_output_shapes
:@:(($
"
_output_shapes
:@@: )

_output_shapes
:@:(*$
"
_output_shapes
:@@: +

_output_shapes
:@:$, 

_output_shapes

:: -

_output_shapes
::%.!

_output_shapes
:	?: /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::(2$
"
_output_shapes
:@: 3

_output_shapes
:@:(4$
"
_output_shapes
:@@: 5

_output_shapes
:@:(6$
"
_output_shapes
:@@: 7

_output_shapes
:@:(8$
"
_output_shapes
:@@: 9

_output_shapes
:@:(:$
"
_output_shapes
:@@: ;

_output_shapes
:@:(<$
"
_output_shapes
:@@: =

_output_shapes
:@:(>$
"
_output_shapes
:@@: ?

_output_shapes
:@:$@ 

_output_shapes

:: A

_output_shapes
::%B!

_output_shapes
:	?: C

_output_shapes
::$D 

_output_shapes

:: E

_output_shapes
::F

_output_shapes
: 
ͱ
?
C__inference_model_3_layer_call_and_return_conditional_losses_231323
inputs_0
inputs_19
5conv1d_21_conv1d_expanddims_1_readvariableop_resource-
)conv1d_21_biasadd_readvariableop_resource9
5conv1d_22_conv1d_expanddims_1_readvariableop_resource-
)conv1d_22_biasadd_readvariableop_resource9
5conv1d_23_conv1d_expanddims_1_readvariableop_resource-
)conv1d_23_biasadd_readvariableop_resource9
5conv1d_24_conv1d_expanddims_1_readvariableop_resource-
)conv1d_24_biasadd_readvariableop_resource9
5conv1d_25_conv1d_expanddims_1_readvariableop_resource-
)conv1d_25_biasadd_readvariableop_resource9
5conv1d_26_conv1d_expanddims_1_readvariableop_resource-
)conv1d_26_biasadd_readvariableop_resource9
5conv1d_27_conv1d_expanddims_1_readvariableop_resource-
)conv1d_27_biasadd_readvariableop_resource-
)dense_9_mlcmatmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource.
*dense_10_mlcmatmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource.
*dense_11_mlcmatmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity?? conv1d_21/BiasAdd/ReadVariableOp?,conv1d_21/conv1d/ExpandDims_1/ReadVariableOp? conv1d_22/BiasAdd/ReadVariableOp?,conv1d_22/conv1d/ExpandDims_1/ReadVariableOp? conv1d_23/BiasAdd/ReadVariableOp?,conv1d_23/conv1d/ExpandDims_1/ReadVariableOp? conv1d_24/BiasAdd/ReadVariableOp?,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp? conv1d_25/BiasAdd/ReadVariableOp?,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp? conv1d_26/BiasAdd/ReadVariableOp?,conv1d_26/conv1d/ExpandDims_1/ReadVariableOp? conv1d_27/BiasAdd/ReadVariableOp?,conv1d_27/conv1d/ExpandDims_1/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?!dense_10/MLCMatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?!dense_11/MLCMatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp? dense_9/MLCMatMul/ReadVariableOp?
conv1d_21/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_21/Pad/paddings?
conv1d_21/PadPadinputs_0conv1d_21/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
conv1d_21/Pad?
conv1d_21/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_21/conv1d/ExpandDims/dim?
conv1d_21/conv1d/ExpandDims
ExpandDimsconv1d_21/Pad:output:0(conv1d_21/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_21/conv1d/ExpandDims?
,conv1d_21/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_21_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_21/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_21/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_21/conv1d/ExpandDims_1/dim?
conv1d_21/conv1d/ExpandDims_1
ExpandDims4conv1d_21/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_21/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_21/conv1d/ExpandDims_1?
conv1d_21/conv1d	MLCConv2D$conv1d_21/conv1d/ExpandDims:output:0&conv1d_21/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_21/conv1d?
conv1d_21/conv1d/SqueezeSqueezeconv1d_21/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_21/conv1d/Squeeze?
 conv1d_21/BiasAdd/ReadVariableOpReadVariableOp)conv1d_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_21/BiasAdd/ReadVariableOp?
conv1d_21/BiasAddBiasAdd!conv1d_21/conv1d/Squeeze:output:0(conv1d_21/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_21/BiasAddz
conv1d_21/ReluReluconv1d_21/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_21/Relu?
conv1d_22/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_22/Pad/paddings?
conv1d_22/PadPadconv1d_21/Relu:activations:0conv1d_22/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_22/Pad?
conv1d_22/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_22/conv1d/ExpandDims/dim?
conv1d_22/conv1d/ExpandDims
ExpandDimsconv1d_22/Pad:output:0(conv1d_22/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_22/conv1d/ExpandDims?
,conv1d_22/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_22_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_22/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_22/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_22/conv1d/ExpandDims_1/dim?
conv1d_22/conv1d/ExpandDims_1
ExpandDims4conv1d_22/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_22/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_22/conv1d/ExpandDims_1?
conv1d_22/conv1d	MLCConv2D$conv1d_22/conv1d/ExpandDims:output:0&conv1d_22/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_22/conv1d?
conv1d_22/conv1d/SqueezeSqueezeconv1d_22/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_22/conv1d/Squeeze?
 conv1d_22/BiasAdd/ReadVariableOpReadVariableOp)conv1d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_22/BiasAdd/ReadVariableOp?
conv1d_22/BiasAddBiasAdd!conv1d_22/conv1d/Squeeze:output:0(conv1d_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_22/BiasAddz
conv1d_22/ReluReluconv1d_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_22/Relu?
conv1d_23/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_23/Pad/paddings?
conv1d_23/PadPadconv1d_22/Relu:activations:0conv1d_23/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_23/Pad?
conv1d_23/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_23/conv1d/ExpandDims/dim?
conv1d_23/conv1d/ExpandDims
ExpandDimsconv1d_23/Pad:output:0(conv1d_23/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_23/conv1d/ExpandDims?
,conv1d_23/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_23_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_23/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_23/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_23/conv1d/ExpandDims_1/dim?
conv1d_23/conv1d/ExpandDims_1
ExpandDims4conv1d_23/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_23/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_23/conv1d/ExpandDims_1?
conv1d_23/conv1d	MLCConv2D$conv1d_23/conv1d/ExpandDims:output:0&conv1d_23/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_23/conv1d?
conv1d_23/conv1d/SqueezeSqueezeconv1d_23/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_23/conv1d/Squeeze?
 conv1d_23/BiasAdd/ReadVariableOpReadVariableOp)conv1d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_23/BiasAdd/ReadVariableOp?
conv1d_23/BiasAddBiasAdd!conv1d_23/conv1d/Squeeze:output:0(conv1d_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_23/BiasAddz
conv1d_23/ReluReluconv1d_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_23/Relu?
conv1d_24/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_24/Pad/paddings?
conv1d_24/PadPadconv1d_23/Relu:activations:0conv1d_24/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_24/Pad?
conv1d_24/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_24/conv1d/ExpandDims/dim?
conv1d_24/conv1d/ExpandDims
ExpandDimsconv1d_24/Pad:output:0(conv1d_24/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_24/conv1d/ExpandDims?
,conv1d_24/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_24_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_24/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_24/conv1d/ExpandDims_1/dim?
conv1d_24/conv1d/ExpandDims_1
ExpandDims4conv1d_24/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_24/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_24/conv1d/ExpandDims_1?
conv1d_24/conv1d	MLCConv2D$conv1d_24/conv1d/ExpandDims:output:0&conv1d_24/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_24/conv1d?
conv1d_24/conv1d/SqueezeSqueezeconv1d_24/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_24/conv1d/Squeeze?
 conv1d_24/BiasAdd/ReadVariableOpReadVariableOp)conv1d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_24/BiasAdd/ReadVariableOp?
conv1d_24/BiasAddBiasAdd!conv1d_24/conv1d/Squeeze:output:0(conv1d_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_24/BiasAddz
conv1d_24/ReluReluconv1d_24/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_24/Relu?
conv1d_25/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_25/Pad/paddings?
conv1d_25/PadPadconv1d_24/Relu:activations:0conv1d_25/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_25/Pad?
conv1d_25/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_25/conv1d/ExpandDims/dim?
conv1d_25/conv1d/ExpandDims
ExpandDimsconv1d_25/Pad:output:0(conv1d_25/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_25/conv1d/ExpandDims?
,conv1d_25/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_25_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_25/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_25/conv1d/ExpandDims_1/dim?
conv1d_25/conv1d/ExpandDims_1
ExpandDims4conv1d_25/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_25/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_25/conv1d/ExpandDims_1?
conv1d_25/conv1d	MLCConv2D$conv1d_25/conv1d/ExpandDims:output:0&conv1d_25/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_25/conv1d?
conv1d_25/conv1d/SqueezeSqueezeconv1d_25/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_25/conv1d/Squeeze?
 conv1d_25/BiasAdd/ReadVariableOpReadVariableOp)conv1d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_25/BiasAdd/ReadVariableOp?
conv1d_25/BiasAddBiasAdd!conv1d_25/conv1d/Squeeze:output:0(conv1d_25/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_25/BiasAddz
conv1d_25/ReluReluconv1d_25/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_25/Relu?
conv1d_26/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_26/Pad/paddings?
conv1d_26/PadPadconv1d_25/Relu:activations:0conv1d_26/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_26/Pad?
conv1d_26/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_26/conv1d/ExpandDims/dim?
conv1d_26/conv1d/ExpandDims
ExpandDimsconv1d_26/Pad:output:0(conv1d_26/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_26/conv1d/ExpandDims?
,conv1d_26/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_26_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_26/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_26/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_26/conv1d/ExpandDims_1/dim?
conv1d_26/conv1d/ExpandDims_1
ExpandDims4conv1d_26/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_26/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_26/conv1d/ExpandDims_1?
conv1d_26/conv1d	MLCConv2D$conv1d_26/conv1d/ExpandDims:output:0&conv1d_26/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_26/conv1d?
conv1d_26/conv1d/SqueezeSqueezeconv1d_26/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_26/conv1d/Squeeze?
 conv1d_26/BiasAdd/ReadVariableOpReadVariableOp)conv1d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_26/BiasAdd/ReadVariableOp?
conv1d_26/BiasAddBiasAdd!conv1d_26/conv1d/Squeeze:output:0(conv1d_26/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_26/BiasAddz
conv1d_26/ReluReluconv1d_26/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_26/Relu?
conv1d_27/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_27/Pad/paddings?
conv1d_27/PadPadconv1d_26/Relu:activations:0conv1d_27/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_27/Pad?
conv1d_27/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_27/conv1d/ExpandDims/dim?
conv1d_27/conv1d/ExpandDims
ExpandDimsconv1d_27/Pad:output:0(conv1d_27/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_27/conv1d/ExpandDims?
,conv1d_27/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_27_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_27/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_27/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_27/conv1d/ExpandDims_1/dim?
conv1d_27/conv1d/ExpandDims_1
ExpandDims4conv1d_27/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_27/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_27/conv1d/ExpandDims_1?
conv1d_27/conv1d	MLCConv2D$conv1d_27/conv1d/ExpandDims:output:0&conv1d_27/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_27/conv1d?
conv1d_27/conv1d/SqueezeSqueezeconv1d_27/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_27/conv1d/Squeeze?
 conv1d_27/BiasAdd/ReadVariableOpReadVariableOp)conv1d_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_27/BiasAdd/ReadVariableOp?
conv1d_27/BiasAddBiasAdd!conv1d_27/conv1d/Squeeze:output:0(conv1d_27/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_27/BiasAddz
conv1d_27/ReluReluconv1d_27/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_27/Relu?
 dense_9/MLCMatMul/ReadVariableOpReadVariableOp)dense_9_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_9/MLCMatMul/ReadVariableOp?
dense_9/MLCMatMul	MLCMatMulinputs_1(dense_9/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MLCMatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MLCMatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_9/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_3/Const?
flatten_3/ReshapeReshapeconv1d_27/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis?
concatenate_3/concatConcatV2dense_9/Relu:activations:0flatten_3/Reshape:output:0"concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_3/concat?
!dense_10/MLCMatMul/ReadVariableOpReadVariableOp*dense_10_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_10/MLCMatMul/ReadVariableOp?
dense_10/MLCMatMul	MLCMatMulconcatenate_3/concat:output:0)dense_10/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/MLCMatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MLCMatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_10/Relu?
!dense_11/MLCMatMul/ReadVariableOpReadVariableOp*dense_11_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_11/MLCMatMul/ReadVariableOp?
dense_11/MLCMatMul	MLCMatMuldense_10/Relu:activations:0)dense_11/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/MLCMatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MLCMatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_11/Softmax?
IdentityIdentitydense_11/Softmax:softmax:0!^conv1d_21/BiasAdd/ReadVariableOp-^conv1d_21/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_22/BiasAdd/ReadVariableOp-^conv1d_22/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_23/BiasAdd/ReadVariableOp-^conv1d_23/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_24/BiasAdd/ReadVariableOp-^conv1d_24/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_25/BiasAdd/ReadVariableOp-^conv1d_25/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_26/BiasAdd/ReadVariableOp-^conv1d_26/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_27/BiasAdd/ReadVariableOp-^conv1d_27/conv1d/ExpandDims_1/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/MLCMatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/MLCMatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2D
 conv1d_21/BiasAdd/ReadVariableOp conv1d_21/BiasAdd/ReadVariableOp2\
,conv1d_21/conv1d/ExpandDims_1/ReadVariableOp,conv1d_21/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_22/BiasAdd/ReadVariableOp conv1d_22/BiasAdd/ReadVariableOp2\
,conv1d_22/conv1d/ExpandDims_1/ReadVariableOp,conv1d_22/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_23/BiasAdd/ReadVariableOp conv1d_23/BiasAdd/ReadVariableOp2\
,conv1d_23/conv1d/ExpandDims_1/ReadVariableOp,conv1d_23/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_24/BiasAdd/ReadVariableOp conv1d_24/BiasAdd/ReadVariableOp2\
,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_25/BiasAdd/ReadVariableOp conv1d_25/BiasAdd/ReadVariableOp2\
,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_26/BiasAdd/ReadVariableOp conv1d_26/BiasAdd/ReadVariableOp2\
,conv1d_26/conv1d/ExpandDims_1/ReadVariableOp,conv1d_26/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_27/BiasAdd/ReadVariableOp conv1d_27/BiasAdd/ReadVariableOp2\
,conv1d_27/conv1d/ExpandDims_1/ReadVariableOp,conv1d_27/conv1d/ExpandDims_1/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/MLCMatMul/ReadVariableOp!dense_10/MLCMatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/MLCMatMul/ReadVariableOp!dense_11/MLCMatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/MLCMatMul/ReadVariableOp dense_9/MLCMatMul/ReadVariableOp:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_230668

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_231630

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_230392
conv
catA
=model_3_conv1d_21_conv1d_expanddims_1_readvariableop_resource5
1model_3_conv1d_21_biasadd_readvariableop_resourceA
=model_3_conv1d_22_conv1d_expanddims_1_readvariableop_resource5
1model_3_conv1d_22_biasadd_readvariableop_resourceA
=model_3_conv1d_23_conv1d_expanddims_1_readvariableop_resource5
1model_3_conv1d_23_biasadd_readvariableop_resourceA
=model_3_conv1d_24_conv1d_expanddims_1_readvariableop_resource5
1model_3_conv1d_24_biasadd_readvariableop_resourceA
=model_3_conv1d_25_conv1d_expanddims_1_readvariableop_resource5
1model_3_conv1d_25_biasadd_readvariableop_resourceA
=model_3_conv1d_26_conv1d_expanddims_1_readvariableop_resource5
1model_3_conv1d_26_biasadd_readvariableop_resourceA
=model_3_conv1d_27_conv1d_expanddims_1_readvariableop_resource5
1model_3_conv1d_27_biasadd_readvariableop_resource5
1model_3_dense_9_mlcmatmul_readvariableop_resource3
/model_3_dense_9_biasadd_readvariableop_resource6
2model_3_dense_10_mlcmatmul_readvariableop_resource4
0model_3_dense_10_biasadd_readvariableop_resource6
2model_3_dense_11_mlcmatmul_readvariableop_resource4
0model_3_dense_11_biasadd_readvariableop_resource
identity??(model_3/conv1d_21/BiasAdd/ReadVariableOp?4model_3/conv1d_21/conv1d/ExpandDims_1/ReadVariableOp?(model_3/conv1d_22/BiasAdd/ReadVariableOp?4model_3/conv1d_22/conv1d/ExpandDims_1/ReadVariableOp?(model_3/conv1d_23/BiasAdd/ReadVariableOp?4model_3/conv1d_23/conv1d/ExpandDims_1/ReadVariableOp?(model_3/conv1d_24/BiasAdd/ReadVariableOp?4model_3/conv1d_24/conv1d/ExpandDims_1/ReadVariableOp?(model_3/conv1d_25/BiasAdd/ReadVariableOp?4model_3/conv1d_25/conv1d/ExpandDims_1/ReadVariableOp?(model_3/conv1d_26/BiasAdd/ReadVariableOp?4model_3/conv1d_26/conv1d/ExpandDims_1/ReadVariableOp?(model_3/conv1d_27/BiasAdd/ReadVariableOp?4model_3/conv1d_27/conv1d/ExpandDims_1/ReadVariableOp?'model_3/dense_10/BiasAdd/ReadVariableOp?)model_3/dense_10/MLCMatMul/ReadVariableOp?'model_3/dense_11/BiasAdd/ReadVariableOp?)model_3/dense_11/MLCMatMul/ReadVariableOp?&model_3/dense_9/BiasAdd/ReadVariableOp?(model_3/dense_9/MLCMatMul/ReadVariableOp?
model_3/conv1d_21/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_3/conv1d_21/Pad/paddings?
model_3/conv1d_21/PadPadconv'model_3/conv1d_21/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
model_3/conv1d_21/Pad?
'model_3/conv1d_21/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_3/conv1d_21/conv1d/ExpandDims/dim?
#model_3/conv1d_21/conv1d/ExpandDims
ExpandDimsmodel_3/conv1d_21/Pad:output:00model_3/conv1d_21/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2%
#model_3/conv1d_21/conv1d/ExpandDims?
4model_3/conv1d_21/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_3_conv1d_21_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype026
4model_3/conv1d_21/conv1d/ExpandDims_1/ReadVariableOp?
)model_3/conv1d_21/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_3/conv1d_21/conv1d/ExpandDims_1/dim?
%model_3/conv1d_21/conv1d/ExpandDims_1
ExpandDims<model_3/conv1d_21/conv1d/ExpandDims_1/ReadVariableOp:value:02model_3/conv1d_21/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2'
%model_3/conv1d_21/conv1d/ExpandDims_1?
model_3/conv1d_21/conv1d	MLCConv2D,model_3/conv1d_21/conv1d/ExpandDims:output:0.model_3/conv1d_21/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_3/conv1d_21/conv1d?
 model_3/conv1d_21/conv1d/SqueezeSqueeze!model_3/conv1d_21/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_3/conv1d_21/conv1d/Squeeze?
(model_3/conv1d_21/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv1d_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv1d_21/BiasAdd/ReadVariableOp?
model_3/conv1d_21/BiasAddBiasAdd)model_3/conv1d_21/conv1d/Squeeze:output:00model_3/conv1d_21/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_21/BiasAdd?
model_3/conv1d_21/ReluRelu"model_3/conv1d_21/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_21/Relu?
model_3/conv1d_22/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_3/conv1d_22/Pad/paddings?
model_3/conv1d_22/PadPad$model_3/conv1d_21/Relu:activations:0'model_3/conv1d_22/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_22/Pad?
'model_3/conv1d_22/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_3/conv1d_22/conv1d/ExpandDims/dim?
#model_3/conv1d_22/conv1d/ExpandDims
ExpandDimsmodel_3/conv1d_22/Pad:output:00model_3/conv1d_22/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_3/conv1d_22/conv1d/ExpandDims?
4model_3/conv1d_22/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_3_conv1d_22_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_3/conv1d_22/conv1d/ExpandDims_1/ReadVariableOp?
)model_3/conv1d_22/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_3/conv1d_22/conv1d/ExpandDims_1/dim?
%model_3/conv1d_22/conv1d/ExpandDims_1
ExpandDims<model_3/conv1d_22/conv1d/ExpandDims_1/ReadVariableOp:value:02model_3/conv1d_22/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_3/conv1d_22/conv1d/ExpandDims_1?
model_3/conv1d_22/conv1d	MLCConv2D,model_3/conv1d_22/conv1d/ExpandDims:output:0.model_3/conv1d_22/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_3/conv1d_22/conv1d?
 model_3/conv1d_22/conv1d/SqueezeSqueeze!model_3/conv1d_22/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_3/conv1d_22/conv1d/Squeeze?
(model_3/conv1d_22/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv1d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv1d_22/BiasAdd/ReadVariableOp?
model_3/conv1d_22/BiasAddBiasAdd)model_3/conv1d_22/conv1d/Squeeze:output:00model_3/conv1d_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_22/BiasAdd?
model_3/conv1d_22/ReluRelu"model_3/conv1d_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_22/Relu?
model_3/conv1d_23/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_3/conv1d_23/Pad/paddings?
model_3/conv1d_23/PadPad$model_3/conv1d_22/Relu:activations:0'model_3/conv1d_23/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_23/Pad?
'model_3/conv1d_23/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_3/conv1d_23/conv1d/ExpandDims/dim?
#model_3/conv1d_23/conv1d/ExpandDims
ExpandDimsmodel_3/conv1d_23/Pad:output:00model_3/conv1d_23/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_3/conv1d_23/conv1d/ExpandDims?
4model_3/conv1d_23/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_3_conv1d_23_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_3/conv1d_23/conv1d/ExpandDims_1/ReadVariableOp?
)model_3/conv1d_23/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_3/conv1d_23/conv1d/ExpandDims_1/dim?
%model_3/conv1d_23/conv1d/ExpandDims_1
ExpandDims<model_3/conv1d_23/conv1d/ExpandDims_1/ReadVariableOp:value:02model_3/conv1d_23/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_3/conv1d_23/conv1d/ExpandDims_1?
model_3/conv1d_23/conv1d	MLCConv2D,model_3/conv1d_23/conv1d/ExpandDims:output:0.model_3/conv1d_23/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_3/conv1d_23/conv1d?
 model_3/conv1d_23/conv1d/SqueezeSqueeze!model_3/conv1d_23/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_3/conv1d_23/conv1d/Squeeze?
(model_3/conv1d_23/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv1d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv1d_23/BiasAdd/ReadVariableOp?
model_3/conv1d_23/BiasAddBiasAdd)model_3/conv1d_23/conv1d/Squeeze:output:00model_3/conv1d_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_23/BiasAdd?
model_3/conv1d_23/ReluRelu"model_3/conv1d_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_23/Relu?
model_3/conv1d_24/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_3/conv1d_24/Pad/paddings?
model_3/conv1d_24/PadPad$model_3/conv1d_23/Relu:activations:0'model_3/conv1d_24/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_24/Pad?
'model_3/conv1d_24/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_3/conv1d_24/conv1d/ExpandDims/dim?
#model_3/conv1d_24/conv1d/ExpandDims
ExpandDimsmodel_3/conv1d_24/Pad:output:00model_3/conv1d_24/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_3/conv1d_24/conv1d/ExpandDims?
4model_3/conv1d_24/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_3_conv1d_24_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_3/conv1d_24/conv1d/ExpandDims_1/ReadVariableOp?
)model_3/conv1d_24/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_3/conv1d_24/conv1d/ExpandDims_1/dim?
%model_3/conv1d_24/conv1d/ExpandDims_1
ExpandDims<model_3/conv1d_24/conv1d/ExpandDims_1/ReadVariableOp:value:02model_3/conv1d_24/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_3/conv1d_24/conv1d/ExpandDims_1?
model_3/conv1d_24/conv1d	MLCConv2D,model_3/conv1d_24/conv1d/ExpandDims:output:0.model_3/conv1d_24/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_3/conv1d_24/conv1d?
 model_3/conv1d_24/conv1d/SqueezeSqueeze!model_3/conv1d_24/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_3/conv1d_24/conv1d/Squeeze?
(model_3/conv1d_24/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv1d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv1d_24/BiasAdd/ReadVariableOp?
model_3/conv1d_24/BiasAddBiasAdd)model_3/conv1d_24/conv1d/Squeeze:output:00model_3/conv1d_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_24/BiasAdd?
model_3/conv1d_24/ReluRelu"model_3/conv1d_24/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_24/Relu?
model_3/conv1d_25/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_3/conv1d_25/Pad/paddings?
model_3/conv1d_25/PadPad$model_3/conv1d_24/Relu:activations:0'model_3/conv1d_25/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_25/Pad?
'model_3/conv1d_25/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_3/conv1d_25/conv1d/ExpandDims/dim?
#model_3/conv1d_25/conv1d/ExpandDims
ExpandDimsmodel_3/conv1d_25/Pad:output:00model_3/conv1d_25/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_3/conv1d_25/conv1d/ExpandDims?
4model_3/conv1d_25/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_3_conv1d_25_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_3/conv1d_25/conv1d/ExpandDims_1/ReadVariableOp?
)model_3/conv1d_25/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_3/conv1d_25/conv1d/ExpandDims_1/dim?
%model_3/conv1d_25/conv1d/ExpandDims_1
ExpandDims<model_3/conv1d_25/conv1d/ExpandDims_1/ReadVariableOp:value:02model_3/conv1d_25/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_3/conv1d_25/conv1d/ExpandDims_1?
model_3/conv1d_25/conv1d	MLCConv2D,model_3/conv1d_25/conv1d/ExpandDims:output:0.model_3/conv1d_25/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_3/conv1d_25/conv1d?
 model_3/conv1d_25/conv1d/SqueezeSqueeze!model_3/conv1d_25/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_3/conv1d_25/conv1d/Squeeze?
(model_3/conv1d_25/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv1d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv1d_25/BiasAdd/ReadVariableOp?
model_3/conv1d_25/BiasAddBiasAdd)model_3/conv1d_25/conv1d/Squeeze:output:00model_3/conv1d_25/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_25/BiasAdd?
model_3/conv1d_25/ReluRelu"model_3/conv1d_25/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_25/Relu?
model_3/conv1d_26/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_3/conv1d_26/Pad/paddings?
model_3/conv1d_26/PadPad$model_3/conv1d_25/Relu:activations:0'model_3/conv1d_26/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_26/Pad?
'model_3/conv1d_26/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_3/conv1d_26/conv1d/ExpandDims/dim?
#model_3/conv1d_26/conv1d/ExpandDims
ExpandDimsmodel_3/conv1d_26/Pad:output:00model_3/conv1d_26/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_3/conv1d_26/conv1d/ExpandDims?
4model_3/conv1d_26/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_3_conv1d_26_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_3/conv1d_26/conv1d/ExpandDims_1/ReadVariableOp?
)model_3/conv1d_26/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_3/conv1d_26/conv1d/ExpandDims_1/dim?
%model_3/conv1d_26/conv1d/ExpandDims_1
ExpandDims<model_3/conv1d_26/conv1d/ExpandDims_1/ReadVariableOp:value:02model_3/conv1d_26/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_3/conv1d_26/conv1d/ExpandDims_1?
model_3/conv1d_26/conv1d	MLCConv2D,model_3/conv1d_26/conv1d/ExpandDims:output:0.model_3/conv1d_26/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_3/conv1d_26/conv1d?
 model_3/conv1d_26/conv1d/SqueezeSqueeze!model_3/conv1d_26/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_3/conv1d_26/conv1d/Squeeze?
(model_3/conv1d_26/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv1d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv1d_26/BiasAdd/ReadVariableOp?
model_3/conv1d_26/BiasAddBiasAdd)model_3/conv1d_26/conv1d/Squeeze:output:00model_3/conv1d_26/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_26/BiasAdd?
model_3/conv1d_26/ReluRelu"model_3/conv1d_26/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_26/Relu?
model_3/conv1d_27/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_3/conv1d_27/Pad/paddings?
model_3/conv1d_27/PadPad$model_3/conv1d_26/Relu:activations:0'model_3/conv1d_27/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_27/Pad?
'model_3/conv1d_27/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_3/conv1d_27/conv1d/ExpandDims/dim?
#model_3/conv1d_27/conv1d/ExpandDims
ExpandDimsmodel_3/conv1d_27/Pad:output:00model_3/conv1d_27/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_3/conv1d_27/conv1d/ExpandDims?
4model_3/conv1d_27/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_3_conv1d_27_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_3/conv1d_27/conv1d/ExpandDims_1/ReadVariableOp?
)model_3/conv1d_27/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_3/conv1d_27/conv1d/ExpandDims_1/dim?
%model_3/conv1d_27/conv1d/ExpandDims_1
ExpandDims<model_3/conv1d_27/conv1d/ExpandDims_1/ReadVariableOp:value:02model_3/conv1d_27/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_3/conv1d_27/conv1d/ExpandDims_1?
model_3/conv1d_27/conv1d	MLCConv2D,model_3/conv1d_27/conv1d/ExpandDims:output:0.model_3/conv1d_27/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_3/conv1d_27/conv1d?
 model_3/conv1d_27/conv1d/SqueezeSqueeze!model_3/conv1d_27/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_3/conv1d_27/conv1d/Squeeze?
(model_3/conv1d_27/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv1d_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv1d_27/BiasAdd/ReadVariableOp?
model_3/conv1d_27/BiasAddBiasAdd)model_3/conv1d_27/conv1d/Squeeze:output:00model_3/conv1d_27/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_27/BiasAdd?
model_3/conv1d_27/ReluRelu"model_3/conv1d_27/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_3/conv1d_27/Relu?
(model_3/dense_9/MLCMatMul/ReadVariableOpReadVariableOp1model_3_dense_9_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(model_3/dense_9/MLCMatMul/ReadVariableOp?
model_3/dense_9/MLCMatMul	MLCMatMulcat0model_3/dense_9/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_9/MLCMatMul?
&model_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_3/dense_9/BiasAdd/ReadVariableOp?
model_3/dense_9/BiasAddBiasAdd#model_3/dense_9/MLCMatMul:product:0.model_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_9/BiasAdd?
model_3/dense_9/ReluRelu model_3/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/dense_9/Relu?
model_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model_3/flatten_3/Const?
model_3/flatten_3/ReshapeReshape$model_3/conv1d_27/Relu:activations:0 model_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
model_3/flatten_3/Reshape?
!model_3/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_3/concatenate_3/concat/axis?
model_3/concatenate_3/concatConcatV2"model_3/dense_9/Relu:activations:0"model_3/flatten_3/Reshape:output:0*model_3/concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model_3/concatenate_3/concat?
)model_3/dense_10/MLCMatMul/ReadVariableOpReadVariableOp2model_3_dense_10_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)model_3/dense_10/MLCMatMul/ReadVariableOp?
model_3/dense_10/MLCMatMul	MLCMatMul%model_3/concatenate_3/concat:output:01model_3/dense_10/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_10/MLCMatMul?
'model_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_3/dense_10/BiasAdd/ReadVariableOp?
model_3/dense_10/BiasAddBiasAdd$model_3/dense_10/MLCMatMul:product:0/model_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_10/BiasAdd?
model_3/dense_10/ReluRelu!model_3/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/dense_10/Relu?
)model_3/dense_11/MLCMatMul/ReadVariableOpReadVariableOp2model_3_dense_11_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)model_3/dense_11/MLCMatMul/ReadVariableOp?
model_3/dense_11/MLCMatMul	MLCMatMul#model_3/dense_10/Relu:activations:01model_3/dense_11/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_11/MLCMatMul?
'model_3/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_3/dense_11/BiasAdd/ReadVariableOp?
model_3/dense_11/BiasAddBiasAdd$model_3/dense_11/MLCMatMul:product:0/model_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_11/BiasAdd?
model_3/dense_11/SoftmaxSoftmax!model_3/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/dense_11/Softmax?
IdentityIdentity"model_3/dense_11/Softmax:softmax:0)^model_3/conv1d_21/BiasAdd/ReadVariableOp5^model_3/conv1d_21/conv1d/ExpandDims_1/ReadVariableOp)^model_3/conv1d_22/BiasAdd/ReadVariableOp5^model_3/conv1d_22/conv1d/ExpandDims_1/ReadVariableOp)^model_3/conv1d_23/BiasAdd/ReadVariableOp5^model_3/conv1d_23/conv1d/ExpandDims_1/ReadVariableOp)^model_3/conv1d_24/BiasAdd/ReadVariableOp5^model_3/conv1d_24/conv1d/ExpandDims_1/ReadVariableOp)^model_3/conv1d_25/BiasAdd/ReadVariableOp5^model_3/conv1d_25/conv1d/ExpandDims_1/ReadVariableOp)^model_3/conv1d_26/BiasAdd/ReadVariableOp5^model_3/conv1d_26/conv1d/ExpandDims_1/ReadVariableOp)^model_3/conv1d_27/BiasAdd/ReadVariableOp5^model_3/conv1d_27/conv1d/ExpandDims_1/ReadVariableOp(^model_3/dense_10/BiasAdd/ReadVariableOp*^model_3/dense_10/MLCMatMul/ReadVariableOp(^model_3/dense_11/BiasAdd/ReadVariableOp*^model_3/dense_11/MLCMatMul/ReadVariableOp'^model_3/dense_9/BiasAdd/ReadVariableOp)^model_3/dense_9/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2T
(model_3/conv1d_21/BiasAdd/ReadVariableOp(model_3/conv1d_21/BiasAdd/ReadVariableOp2l
4model_3/conv1d_21/conv1d/ExpandDims_1/ReadVariableOp4model_3/conv1d_21/conv1d/ExpandDims_1/ReadVariableOp2T
(model_3/conv1d_22/BiasAdd/ReadVariableOp(model_3/conv1d_22/BiasAdd/ReadVariableOp2l
4model_3/conv1d_22/conv1d/ExpandDims_1/ReadVariableOp4model_3/conv1d_22/conv1d/ExpandDims_1/ReadVariableOp2T
(model_3/conv1d_23/BiasAdd/ReadVariableOp(model_3/conv1d_23/BiasAdd/ReadVariableOp2l
4model_3/conv1d_23/conv1d/ExpandDims_1/ReadVariableOp4model_3/conv1d_23/conv1d/ExpandDims_1/ReadVariableOp2T
(model_3/conv1d_24/BiasAdd/ReadVariableOp(model_3/conv1d_24/BiasAdd/ReadVariableOp2l
4model_3/conv1d_24/conv1d/ExpandDims_1/ReadVariableOp4model_3/conv1d_24/conv1d/ExpandDims_1/ReadVariableOp2T
(model_3/conv1d_25/BiasAdd/ReadVariableOp(model_3/conv1d_25/BiasAdd/ReadVariableOp2l
4model_3/conv1d_25/conv1d/ExpandDims_1/ReadVariableOp4model_3/conv1d_25/conv1d/ExpandDims_1/ReadVariableOp2T
(model_3/conv1d_26/BiasAdd/ReadVariableOp(model_3/conv1d_26/BiasAdd/ReadVariableOp2l
4model_3/conv1d_26/conv1d/ExpandDims_1/ReadVariableOp4model_3/conv1d_26/conv1d/ExpandDims_1/ReadVariableOp2T
(model_3/conv1d_27/BiasAdd/ReadVariableOp(model_3/conv1d_27/BiasAdd/ReadVariableOp2l
4model_3/conv1d_27/conv1d/ExpandDims_1/ReadVariableOp4model_3/conv1d_27/conv1d/ExpandDims_1/ReadVariableOp2R
'model_3/dense_10/BiasAdd/ReadVariableOp'model_3/dense_10/BiasAdd/ReadVariableOp2V
)model_3/dense_10/MLCMatMul/ReadVariableOp)model_3/dense_10/MLCMatMul/ReadVariableOp2R
'model_3/dense_11/BiasAdd/ReadVariableOp'model_3/dense_11/BiasAdd/ReadVariableOp2V
)model_3/dense_11/MLCMatMul/ReadVariableOp)model_3/dense_11/MLCMatMul/ReadVariableOp2P
&model_3/dense_9/BiasAdd/ReadVariableOp&model_3/dense_9/BiasAdd/ReadVariableOp2T
(model_3/dense_9/MLCMatMul/ReadVariableOp(model_3/dense_9/MLCMatMul/ReadVariableOp:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
ͱ
?
C__inference_model_3_layer_call_and_return_conditional_losses_231195
inputs_0
inputs_19
5conv1d_21_conv1d_expanddims_1_readvariableop_resource-
)conv1d_21_biasadd_readvariableop_resource9
5conv1d_22_conv1d_expanddims_1_readvariableop_resource-
)conv1d_22_biasadd_readvariableop_resource9
5conv1d_23_conv1d_expanddims_1_readvariableop_resource-
)conv1d_23_biasadd_readvariableop_resource9
5conv1d_24_conv1d_expanddims_1_readvariableop_resource-
)conv1d_24_biasadd_readvariableop_resource9
5conv1d_25_conv1d_expanddims_1_readvariableop_resource-
)conv1d_25_biasadd_readvariableop_resource9
5conv1d_26_conv1d_expanddims_1_readvariableop_resource-
)conv1d_26_biasadd_readvariableop_resource9
5conv1d_27_conv1d_expanddims_1_readvariableop_resource-
)conv1d_27_biasadd_readvariableop_resource-
)dense_9_mlcmatmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource.
*dense_10_mlcmatmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource.
*dense_11_mlcmatmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity?? conv1d_21/BiasAdd/ReadVariableOp?,conv1d_21/conv1d/ExpandDims_1/ReadVariableOp? conv1d_22/BiasAdd/ReadVariableOp?,conv1d_22/conv1d/ExpandDims_1/ReadVariableOp? conv1d_23/BiasAdd/ReadVariableOp?,conv1d_23/conv1d/ExpandDims_1/ReadVariableOp? conv1d_24/BiasAdd/ReadVariableOp?,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp? conv1d_25/BiasAdd/ReadVariableOp?,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp? conv1d_26/BiasAdd/ReadVariableOp?,conv1d_26/conv1d/ExpandDims_1/ReadVariableOp? conv1d_27/BiasAdd/ReadVariableOp?,conv1d_27/conv1d/ExpandDims_1/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?!dense_10/MLCMatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?!dense_11/MLCMatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp? dense_9/MLCMatMul/ReadVariableOp?
conv1d_21/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_21/Pad/paddings?
conv1d_21/PadPadinputs_0conv1d_21/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
conv1d_21/Pad?
conv1d_21/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_21/conv1d/ExpandDims/dim?
conv1d_21/conv1d/ExpandDims
ExpandDimsconv1d_21/Pad:output:0(conv1d_21/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_21/conv1d/ExpandDims?
,conv1d_21/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_21_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_21/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_21/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_21/conv1d/ExpandDims_1/dim?
conv1d_21/conv1d/ExpandDims_1
ExpandDims4conv1d_21/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_21/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_21/conv1d/ExpandDims_1?
conv1d_21/conv1d	MLCConv2D$conv1d_21/conv1d/ExpandDims:output:0&conv1d_21/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_21/conv1d?
conv1d_21/conv1d/SqueezeSqueezeconv1d_21/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_21/conv1d/Squeeze?
 conv1d_21/BiasAdd/ReadVariableOpReadVariableOp)conv1d_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_21/BiasAdd/ReadVariableOp?
conv1d_21/BiasAddBiasAdd!conv1d_21/conv1d/Squeeze:output:0(conv1d_21/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_21/BiasAddz
conv1d_21/ReluReluconv1d_21/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_21/Relu?
conv1d_22/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_22/Pad/paddings?
conv1d_22/PadPadconv1d_21/Relu:activations:0conv1d_22/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_22/Pad?
conv1d_22/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_22/conv1d/ExpandDims/dim?
conv1d_22/conv1d/ExpandDims
ExpandDimsconv1d_22/Pad:output:0(conv1d_22/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_22/conv1d/ExpandDims?
,conv1d_22/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_22_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_22/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_22/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_22/conv1d/ExpandDims_1/dim?
conv1d_22/conv1d/ExpandDims_1
ExpandDims4conv1d_22/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_22/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_22/conv1d/ExpandDims_1?
conv1d_22/conv1d	MLCConv2D$conv1d_22/conv1d/ExpandDims:output:0&conv1d_22/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_22/conv1d?
conv1d_22/conv1d/SqueezeSqueezeconv1d_22/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_22/conv1d/Squeeze?
 conv1d_22/BiasAdd/ReadVariableOpReadVariableOp)conv1d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_22/BiasAdd/ReadVariableOp?
conv1d_22/BiasAddBiasAdd!conv1d_22/conv1d/Squeeze:output:0(conv1d_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_22/BiasAddz
conv1d_22/ReluReluconv1d_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_22/Relu?
conv1d_23/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_23/Pad/paddings?
conv1d_23/PadPadconv1d_22/Relu:activations:0conv1d_23/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_23/Pad?
conv1d_23/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_23/conv1d/ExpandDims/dim?
conv1d_23/conv1d/ExpandDims
ExpandDimsconv1d_23/Pad:output:0(conv1d_23/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_23/conv1d/ExpandDims?
,conv1d_23/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_23_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_23/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_23/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_23/conv1d/ExpandDims_1/dim?
conv1d_23/conv1d/ExpandDims_1
ExpandDims4conv1d_23/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_23/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_23/conv1d/ExpandDims_1?
conv1d_23/conv1d	MLCConv2D$conv1d_23/conv1d/ExpandDims:output:0&conv1d_23/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_23/conv1d?
conv1d_23/conv1d/SqueezeSqueezeconv1d_23/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_23/conv1d/Squeeze?
 conv1d_23/BiasAdd/ReadVariableOpReadVariableOp)conv1d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_23/BiasAdd/ReadVariableOp?
conv1d_23/BiasAddBiasAdd!conv1d_23/conv1d/Squeeze:output:0(conv1d_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_23/BiasAddz
conv1d_23/ReluReluconv1d_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_23/Relu?
conv1d_24/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_24/Pad/paddings?
conv1d_24/PadPadconv1d_23/Relu:activations:0conv1d_24/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_24/Pad?
conv1d_24/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_24/conv1d/ExpandDims/dim?
conv1d_24/conv1d/ExpandDims
ExpandDimsconv1d_24/Pad:output:0(conv1d_24/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_24/conv1d/ExpandDims?
,conv1d_24/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_24_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_24/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_24/conv1d/ExpandDims_1/dim?
conv1d_24/conv1d/ExpandDims_1
ExpandDims4conv1d_24/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_24/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_24/conv1d/ExpandDims_1?
conv1d_24/conv1d	MLCConv2D$conv1d_24/conv1d/ExpandDims:output:0&conv1d_24/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_24/conv1d?
conv1d_24/conv1d/SqueezeSqueezeconv1d_24/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_24/conv1d/Squeeze?
 conv1d_24/BiasAdd/ReadVariableOpReadVariableOp)conv1d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_24/BiasAdd/ReadVariableOp?
conv1d_24/BiasAddBiasAdd!conv1d_24/conv1d/Squeeze:output:0(conv1d_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_24/BiasAddz
conv1d_24/ReluReluconv1d_24/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_24/Relu?
conv1d_25/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_25/Pad/paddings?
conv1d_25/PadPadconv1d_24/Relu:activations:0conv1d_25/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_25/Pad?
conv1d_25/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_25/conv1d/ExpandDims/dim?
conv1d_25/conv1d/ExpandDims
ExpandDimsconv1d_25/Pad:output:0(conv1d_25/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_25/conv1d/ExpandDims?
,conv1d_25/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_25_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_25/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_25/conv1d/ExpandDims_1/dim?
conv1d_25/conv1d/ExpandDims_1
ExpandDims4conv1d_25/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_25/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_25/conv1d/ExpandDims_1?
conv1d_25/conv1d	MLCConv2D$conv1d_25/conv1d/ExpandDims:output:0&conv1d_25/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_25/conv1d?
conv1d_25/conv1d/SqueezeSqueezeconv1d_25/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_25/conv1d/Squeeze?
 conv1d_25/BiasAdd/ReadVariableOpReadVariableOp)conv1d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_25/BiasAdd/ReadVariableOp?
conv1d_25/BiasAddBiasAdd!conv1d_25/conv1d/Squeeze:output:0(conv1d_25/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_25/BiasAddz
conv1d_25/ReluReluconv1d_25/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_25/Relu?
conv1d_26/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_26/Pad/paddings?
conv1d_26/PadPadconv1d_25/Relu:activations:0conv1d_26/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_26/Pad?
conv1d_26/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_26/conv1d/ExpandDims/dim?
conv1d_26/conv1d/ExpandDims
ExpandDimsconv1d_26/Pad:output:0(conv1d_26/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_26/conv1d/ExpandDims?
,conv1d_26/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_26_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_26/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_26/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_26/conv1d/ExpandDims_1/dim?
conv1d_26/conv1d/ExpandDims_1
ExpandDims4conv1d_26/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_26/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_26/conv1d/ExpandDims_1?
conv1d_26/conv1d	MLCConv2D$conv1d_26/conv1d/ExpandDims:output:0&conv1d_26/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_26/conv1d?
conv1d_26/conv1d/SqueezeSqueezeconv1d_26/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_26/conv1d/Squeeze?
 conv1d_26/BiasAdd/ReadVariableOpReadVariableOp)conv1d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_26/BiasAdd/ReadVariableOp?
conv1d_26/BiasAddBiasAdd!conv1d_26/conv1d/Squeeze:output:0(conv1d_26/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_26/BiasAddz
conv1d_26/ReluReluconv1d_26/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_26/Relu?
conv1d_27/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_27/Pad/paddings?
conv1d_27/PadPadconv1d_26/Relu:activations:0conv1d_27/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_27/Pad?
conv1d_27/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_27/conv1d/ExpandDims/dim?
conv1d_27/conv1d/ExpandDims
ExpandDimsconv1d_27/Pad:output:0(conv1d_27/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_27/conv1d/ExpandDims?
,conv1d_27/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_27_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_27/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_27/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_27/conv1d/ExpandDims_1/dim?
conv1d_27/conv1d/ExpandDims_1
ExpandDims4conv1d_27/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_27/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_27/conv1d/ExpandDims_1?
conv1d_27/conv1d	MLCConv2D$conv1d_27/conv1d/ExpandDims:output:0&conv1d_27/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_27/conv1d?
conv1d_27/conv1d/SqueezeSqueezeconv1d_27/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_27/conv1d/Squeeze?
 conv1d_27/BiasAdd/ReadVariableOpReadVariableOp)conv1d_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_27/BiasAdd/ReadVariableOp?
conv1d_27/BiasAddBiasAdd!conv1d_27/conv1d/Squeeze:output:0(conv1d_27/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_27/BiasAddz
conv1d_27/ReluReluconv1d_27/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_27/Relu?
 dense_9/MLCMatMul/ReadVariableOpReadVariableOp)dense_9_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_9/MLCMatMul/ReadVariableOp?
dense_9/MLCMatMul	MLCMatMulinputs_1(dense_9/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MLCMatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MLCMatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_9/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_3/Const?
flatten_3/ReshapeReshapeconv1d_27/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis?
concatenate_3/concatConcatV2dense_9/Relu:activations:0flatten_3/Reshape:output:0"concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_3/concat?
!dense_10/MLCMatMul/ReadVariableOpReadVariableOp*dense_10_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_10/MLCMatMul/ReadVariableOp?
dense_10/MLCMatMul	MLCMatMulconcatenate_3/concat:output:0)dense_10/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/MLCMatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MLCMatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_10/Relu?
!dense_11/MLCMatMul/ReadVariableOpReadVariableOp*dense_11_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_11/MLCMatMul/ReadVariableOp?
dense_11/MLCMatMul	MLCMatMuldense_10/Relu:activations:0)dense_11/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/MLCMatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MLCMatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_11/Softmax?
IdentityIdentitydense_11/Softmax:softmax:0!^conv1d_21/BiasAdd/ReadVariableOp-^conv1d_21/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_22/BiasAdd/ReadVariableOp-^conv1d_22/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_23/BiasAdd/ReadVariableOp-^conv1d_23/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_24/BiasAdd/ReadVariableOp-^conv1d_24/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_25/BiasAdd/ReadVariableOp-^conv1d_25/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_26/BiasAdd/ReadVariableOp-^conv1d_26/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_27/BiasAdd/ReadVariableOp-^conv1d_27/conv1d/ExpandDims_1/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/MLCMatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/MLCMatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2D
 conv1d_21/BiasAdd/ReadVariableOp conv1d_21/BiasAdd/ReadVariableOp2\
,conv1d_21/conv1d/ExpandDims_1/ReadVariableOp,conv1d_21/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_22/BiasAdd/ReadVariableOp conv1d_22/BiasAdd/ReadVariableOp2\
,conv1d_22/conv1d/ExpandDims_1/ReadVariableOp,conv1d_22/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_23/BiasAdd/ReadVariableOp conv1d_23/BiasAdd/ReadVariableOp2\
,conv1d_23/conv1d/ExpandDims_1/ReadVariableOp,conv1d_23/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_24/BiasAdd/ReadVariableOp conv1d_24/BiasAdd/ReadVariableOp2\
,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_25/BiasAdd/ReadVariableOp conv1d_25/BiasAdd/ReadVariableOp2\
,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_26/BiasAdd/ReadVariableOp conv1d_26/BiasAdd/ReadVariableOp2\
,conv1d_26/conv1d/ExpandDims_1/ReadVariableOp,conv1d_26/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_27/BiasAdd/ReadVariableOp conv1d_27/BiasAdd/ReadVariableOp2\
,conv1d_27/conv1d/ExpandDims_1/ReadVariableOp,conv1d_27/conv1d/ExpandDims_1/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/MLCMatMul/ReadVariableOp!dense_10/MLCMatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/MLCMatMul/ReadVariableOp!dense_11/MLCMatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/MLCMatMul/ReadVariableOp dense_9/MLCMatMul/ReadVariableOp:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
E__inference_conv1d_24_layer_call_and_return_conditional_losses_231514

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOp?
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1d	MLCConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_model_3_layer_call_fn_231369
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_2308652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
D__inference_dense_11_layer_call_and_return_conditional_losses_230730

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
3
cat,
serving_default_cat:0?????????
9
conv1
serving_default_conv:0?????????<
dense_110
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_networkˁ{"class_name": "Functional", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv"}, "name": "conv", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_21", "inbound_nodes": [[["conv", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_22", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_22", "inbound_nodes": [[["conv1d_21", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_23", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_23", "inbound_nodes": [[["conv1d_22", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_24", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_24", "inbound_nodes": [[["conv1d_23", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_25", "inbound_nodes": [[["conv1d_24", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_26", "inbound_nodes": [[["conv1d_25", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cat"}, "name": "cat", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_27", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_27", "inbound_nodes": [[["conv1d_26", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 7, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["cat", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["conv1d_27", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["dense_9", 0, 0, {}], ["flatten_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 19, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}], "input_layers": [["conv", 0, 0], ["cat", 0, 0]], "output_layers": [["dense_11", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 5]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 26, 1]}, {"class_name": "TensorShape", "items": [null, 5]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv"}, "name": "conv", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_21", "inbound_nodes": [[["conv", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_22", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_22", "inbound_nodes": [[["conv1d_21", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_23", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_23", "inbound_nodes": [[["conv1d_22", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_24", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_24", "inbound_nodes": [[["conv1d_23", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_25", "inbound_nodes": [[["conv1d_24", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_26", "inbound_nodes": [[["conv1d_25", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cat"}, "name": "cat", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_27", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_27", "inbound_nodes": [[["conv1d_26", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 7, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["cat", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["conv1d_27", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["dense_9", 0, 0, {}], ["flatten_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 19, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}], "input_layers": [["conv", 0, 0], ["cat", 0, 0]], "output_layers": [["dense_11", 0, 0]]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv"}}
?


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv1D", "name": "conv1d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 1]}}
?	

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_22", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_23", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_24", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

3kernel
4bias
5trainable_variables
6	variables
7regularization_losses
8	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cat", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cat"}}
?	

9kernel
:bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_27", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?

?kernel
@bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 7, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}
?
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 7]}, {"class_name": "TensorShape", "items": [null, 1664]}]}
?

Mkernel
Nbias
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 19, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1671}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1671]}}
?

Skernel
Tbias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 19}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19]}}
?
Yiter

Zbeta_1

[beta_2
	\decay
]learning_ratem?m?m?m?!m?"m?'m?(m?-m?.m?3m?4m?9m?:m??m?@m?Mm?Nm?Sm?Tm?v?v?v?v?!v?"v?'v?(v?-v?.v?3v?4v?9v?:v??v?@v?Mv?Nv?Sv?Tv?"
	optimizer
?
0
1
2
3
!4
"5
'6
(7
-8
.9
310
411
912
:13
?14
@15
M16
N17
S18
T19"
trackable_list_wrapper
?
0
1
2
3
!4
"5
'6
(7
-8
.9
310
411
912
:13
?14
@15
M16
N17
S18
T19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^layer_regularization_losses
_layer_metrics
trainable_variables

`layers
ametrics
	variables
regularization_losses
bnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
&:$@2conv1d_21/kernel
:@2conv1d_21/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
clayer_regularization_losses
dlayer_metrics
trainable_variables

elayers
fmetrics
	variables
regularization_losses
gnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_22/kernel
:@2conv1d_22/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
hlayer_regularization_losses
ilayer_metrics
trainable_variables

jlayers
kmetrics
	variables
regularization_losses
lnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_23/kernel
:@2conv1d_23/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
mlayer_regularization_losses
nlayer_metrics
#trainable_variables

olayers
pmetrics
$	variables
%regularization_losses
qnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_24/kernel
:@2conv1d_24/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
rlayer_regularization_losses
slayer_metrics
)trainable_variables

tlayers
umetrics
*	variables
+regularization_losses
vnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_25/kernel
:@2conv1d_25/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
wlayer_regularization_losses
xlayer_metrics
/trainable_variables

ylayers
zmetrics
0	variables
1regularization_losses
{non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_26/kernel
:@2conv1d_26/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
|layer_regularization_losses
}layer_metrics
5trainable_variables

~layers
metrics
6	variables
7regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_27/kernel
:@2conv1d_27/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
;trainable_variables
?layers
?metrics
<	variables
=regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_9/kernel
:2dense_9/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
Atrainable_variables
?layers
?metrics
B	variables
Cregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
Etrainable_variables
?layers
?metrics
F	variables
Gregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
Itrainable_variables
?layers
?metrics
J	variables
Kregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_10/kernel
:2dense_10/bias
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
Otrainable_variables
?layers
?metrics
P	variables
Qregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_11/kernel
:2dense_11/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
Utrainable_variables
?layers
?metrics
V	variables
Wregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
+:)@2Adam/conv1d_21/kernel/m
!:@2Adam/conv1d_21/bias/m
+:)@@2Adam/conv1d_22/kernel/m
!:@2Adam/conv1d_22/bias/m
+:)@@2Adam/conv1d_23/kernel/m
!:@2Adam/conv1d_23/bias/m
+:)@@2Adam/conv1d_24/kernel/m
!:@2Adam/conv1d_24/bias/m
+:)@@2Adam/conv1d_25/kernel/m
!:@2Adam/conv1d_25/bias/m
+:)@@2Adam/conv1d_26/kernel/m
!:@2Adam/conv1d_26/bias/m
+:)@@2Adam/conv1d_27/kernel/m
!:@2Adam/conv1d_27/bias/m
%:#2Adam/dense_9/kernel/m
:2Adam/dense_9/bias/m
':%	?2Adam/dense_10/kernel/m
 :2Adam/dense_10/bias/m
&:$2Adam/dense_11/kernel/m
 :2Adam/dense_11/bias/m
+:)@2Adam/conv1d_21/kernel/v
!:@2Adam/conv1d_21/bias/v
+:)@@2Adam/conv1d_22/kernel/v
!:@2Adam/conv1d_22/bias/v
+:)@@2Adam/conv1d_23/kernel/v
!:@2Adam/conv1d_23/bias/v
+:)@@2Adam/conv1d_24/kernel/v
!:@2Adam/conv1d_24/bias/v
+:)@@2Adam/conv1d_25/kernel/v
!:@2Adam/conv1d_25/bias/v
+:)@@2Adam/conv1d_26/kernel/v
!:@2Adam/conv1d_26/bias/v
+:)@@2Adam/conv1d_27/kernel/v
!:@2Adam/conv1d_27/bias/v
%:#2Adam/dense_9/kernel/v
:2Adam/dense_9/bias/v
':%	?2Adam/dense_10/kernel/v
 :2Adam/dense_10/bias/v
&:$2Adam/dense_11/kernel/v
 :2Adam/dense_11/bias/v
?2?
!__inference__wrapped_model_230392?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *K?H
F?C
"?
conv?????????
?
cat?????????
?2?
C__inference_model_3_layer_call_and_return_conditional_losses_231195
C__inference_model_3_layer_call_and_return_conditional_losses_231323
C__inference_model_3_layer_call_and_return_conditional_losses_230804
C__inference_model_3_layer_call_and_return_conditional_losses_230747?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_model_3_layer_call_fn_231415
(__inference_model_3_layer_call_fn_231369
(__inference_model_3_layer_call_fn_231011
(__inference_model_3_layer_call_fn_230908?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_conv1d_21_layer_call_and_return_conditional_losses_231433?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv1d_21_layer_call_fn_231442?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv1d_22_layer_call_and_return_conditional_losses_231460?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv1d_22_layer_call_fn_231469?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv1d_23_layer_call_and_return_conditional_losses_231487?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv1d_23_layer_call_fn_231496?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv1d_24_layer_call_and_return_conditional_losses_231514?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv1d_24_layer_call_fn_231523?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv1d_25_layer_call_and_return_conditional_losses_231541?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv1d_25_layer_call_fn_231550?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv1d_26_layer_call_and_return_conditional_losses_231568?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv1d_26_layer_call_fn_231577?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv1d_27_layer_call_and_return_conditional_losses_231595?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv1d_27_layer_call_fn_231604?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_9_layer_call_and_return_conditional_losses_231615?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_9_layer_call_fn_231624?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_3_layer_call_and_return_conditional_losses_231630?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_3_layer_call_fn_231635?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_concatenate_3_layer_call_and_return_conditional_losses_231642?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_concatenate_3_layer_call_fn_231648?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_10_layer_call_and_return_conditional_losses_231659?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_10_layer_call_fn_231668?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_11_layer_call_and_return_conditional_losses_231679?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_11_layer_call_fn_231688?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_231067catconv"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_230392?!"'(-.349:?@MNSTU?R
K?H
F?C
"?
conv?????????
?
cat?????????
? "3?0
.
dense_11"?
dense_11??????????
I__inference_concatenate_3_layer_call_and_return_conditional_losses_231642?[?X
Q?N
L?I
"?
inputs/0?????????
#? 
inputs/1??????????
? "&?#
?
0??????????
? ?
.__inference_concatenate_3_layer_call_fn_231648x[?X
Q?N
L?I
"?
inputs/0?????????
#? 
inputs/1??????????
? "????????????
E__inference_conv1d_21_layer_call_and_return_conditional_losses_231433d3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????@
? ?
*__inference_conv1d_21_layer_call_fn_231442W3?0
)?&
$?!
inputs?????????
? "??????????@?
E__inference_conv1d_22_layer_call_and_return_conditional_losses_231460d3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_22_layer_call_fn_231469W3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_23_layer_call_and_return_conditional_losses_231487d!"3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_23_layer_call_fn_231496W!"3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_24_layer_call_and_return_conditional_losses_231514d'(3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_24_layer_call_fn_231523W'(3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_25_layer_call_and_return_conditional_losses_231541d-.3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_25_layer_call_fn_231550W-.3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_26_layer_call_and_return_conditional_losses_231568d343?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_26_layer_call_fn_231577W343?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_27_layer_call_and_return_conditional_losses_231595d9:3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_27_layer_call_fn_231604W9:3?0
)?&
$?!
inputs?????????@
? "??????????@?
D__inference_dense_10_layer_call_and_return_conditional_losses_231659]MN0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_10_layer_call_fn_231668PMN0?-
&?#
!?
inputs??????????
? "???????????
D__inference_dense_11_layer_call_and_return_conditional_losses_231679\ST/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_11_layer_call_fn_231688OST/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_dense_9_layer_call_and_return_conditional_losses_231615\?@/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_9_layer_call_fn_231624O?@/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_3_layer_call_and_return_conditional_losses_231630]3?0
)?&
$?!
inputs?????????@
? "&?#
?
0??????????
? ~
*__inference_flatten_3_layer_call_fn_231635P3?0
)?&
$?!
inputs?????????@
? "????????????
C__inference_model_3_layer_call_and_return_conditional_losses_230747?!"'(-.349:?@MNST]?Z
S?P
F?C
"?
conv?????????
?
cat?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_model_3_layer_call_and_return_conditional_losses_230804?!"'(-.349:?@MNST]?Z
S?P
F?C
"?
conv?????????
?
cat?????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_3_layer_call_and_return_conditional_losses_231195?!"'(-.349:?@MNSTf?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_model_3_layer_call_and_return_conditional_losses_231323?!"'(-.349:?@MNSTf?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
(__inference_model_3_layer_call_fn_230908?!"'(-.349:?@MNST]?Z
S?P
F?C
"?
conv?????????
?
cat?????????
p

 
? "???????????
(__inference_model_3_layer_call_fn_231011?!"'(-.349:?@MNST]?Z
S?P
F?C
"?
conv?????????
?
cat?????????
p 

 
? "???????????
(__inference_model_3_layer_call_fn_231369?!"'(-.349:?@MNSTf?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
(__inference_model_3_layer_call_fn_231415?!"'(-.349:?@MNSTf?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
$__inference_signature_wrapper_231067?!"'(-.349:?@MNST_?\
? 
U?R
$
cat?
cat?????????
*
conv"?
conv?????????"3?0
.
dense_11"?
dense_11?????????