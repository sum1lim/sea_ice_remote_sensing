??
??
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
 ?"serve*	2.4.0-rc02v1.12.1-44683-gbcaa5ccc43e8??
?
conv1d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv1d_35/kernel
y
$conv1d_35/kernel/Read/ReadVariableOpReadVariableOpconv1d_35/kernel*"
_output_shapes
:@*
dtype0
t
conv1d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_35/bias
m
"conv1d_35/bias/Read/ReadVariableOpReadVariableOpconv1d_35/bias*
_output_shapes
:@*
dtype0
?
conv1d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_36/kernel
y
$conv1d_36/kernel/Read/ReadVariableOpReadVariableOpconv1d_36/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_36/bias
m
"conv1d_36/bias/Read/ReadVariableOpReadVariableOpconv1d_36/bias*
_output_shapes
:@*
dtype0
?
conv1d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_37/kernel
y
$conv1d_37/kernel/Read/ReadVariableOpReadVariableOpconv1d_37/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_37/bias
m
"conv1d_37/bias/Read/ReadVariableOpReadVariableOpconv1d_37/bias*
_output_shapes
:@*
dtype0
?
conv1d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_38/kernel
y
$conv1d_38/kernel/Read/ReadVariableOpReadVariableOpconv1d_38/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_38/bias
m
"conv1d_38/bias/Read/ReadVariableOpReadVariableOpconv1d_38/bias*
_output_shapes
:@*
dtype0
?
conv1d_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_39/kernel
y
$conv1d_39/kernel/Read/ReadVariableOpReadVariableOpconv1d_39/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_39/bias
m
"conv1d_39/bias/Read/ReadVariableOpReadVariableOpconv1d_39/bias*
_output_shapes
:@*
dtype0
?
conv1d_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_40/kernel
y
$conv1d_40/kernel/Read/ReadVariableOpReadVariableOpconv1d_40/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_40/bias
m
"conv1d_40/bias/Read/ReadVariableOpReadVariableOpconv1d_40/bias*
_output_shapes
:@*
dtype0
?
conv1d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_41/kernel
y
$conv1d_41/kernel/Read/ReadVariableOpReadVariableOpconv1d_41/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_41/bias
m
"conv1d_41/bias/Read/ReadVariableOpReadVariableOpconv1d_41/bias*
_output_shapes
:@*
dtype0
{
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_10/kernel
t
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes
:	?*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
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
Adam/conv1d_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv1d_35/kernel/m
?
+Adam/conv1d_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_35/kernel/m*"
_output_shapes
:@*
dtype0
?
Adam/conv1d_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_35/bias/m
{
)Adam/conv1d_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_35/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_36/kernel/m
?
+Adam/conv1d_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_36/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_36/bias/m
{
)Adam/conv1d_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_36/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_37/kernel/m
?
+Adam/conv1d_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_37/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_37/bias/m
{
)Adam/conv1d_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_37/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_38/kernel/m
?
+Adam/conv1d_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_38/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_38/bias/m
{
)Adam/conv1d_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_38/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_39/kernel/m
?
+Adam/conv1d_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_39/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_39/bias/m
{
)Adam/conv1d_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_39/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_40/kernel/m
?
+Adam/conv1d_40/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_40/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_40/bias/m
{
)Adam/conv1d_40/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_40/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_41/kernel/m
?
+Adam/conv1d_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_41/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_41/bias/m
{
)Adam/conv1d_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_41/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_10/kernel/m
?
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_11/kernel/m
?
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1d_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv1d_35/kernel/v
?
+Adam/conv1d_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_35/kernel/v*"
_output_shapes
:@*
dtype0
?
Adam/conv1d_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_35/bias/v
{
)Adam/conv1d_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_35/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_36/kernel/v
?
+Adam/conv1d_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_36/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_36/bias/v
{
)Adam/conv1d_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_36/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_37/kernel/v
?
+Adam/conv1d_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_37/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_37/bias/v
{
)Adam/conv1d_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_37/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_38/kernel/v
?
+Adam/conv1d_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_38/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_38/bias/v
{
)Adam/conv1d_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_38/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_39/kernel/v
?
+Adam/conv1d_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_39/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_39/bias/v
{
)Adam/conv1d_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_39/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_40/kernel/v
?
+Adam/conv1d_40/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_40/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_40/bias/v
{
)Adam/conv1d_40/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_40/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_41/kernel/v
?
+Adam/conv1d_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_41/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_41/bias/v
{
)Adam/conv1d_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_41/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_10/kernel/v
?
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_11/kernel/v
?
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?\
value?\B?\ B?\
?
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
layer_with_weights-6
layer-7
	layer-8

layer_with_weights-7

layer-9
layer-10
layer_with_weights-8
layer-11
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
h

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
h

+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
h

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
h

7kernel
8bias
9trainable_variables
:regularization_losses
;	variables
<	keras_api
R
=trainable_variables
>regularization_losses
?	variables
@	keras_api
h

Akernel
Bbias
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
 
h

Gkernel
Hbias
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
?
Miter

Nbeta_1

Obeta_2
	Pdecay
Qlearning_ratem?m?m?m?m? m?%m?&m?+m?,m?1m?2m?7m?8m?Am?Bm?Gm?Hm?v?v?v?v?v? v?%v?&v?+v?,v?1v?2v?7v?8v?Av?Bv?Gv?Hv?
?
0
1
2
3
4
 5
%6
&7
+8
,9
110
211
712
813
A14
B15
G16
H17
 
?
0
1
2
3
4
 5
%6
&7
+8
,9
110
211
712
813
A14
B15
G16
H17
?
trainable_variables
regularization_losses
Rlayer_regularization_losses
Snon_trainable_variables
Tlayer_metrics
	variables
Umetrics

Vlayers
 
\Z
VARIABLE_VALUEconv1d_35/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_35/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
trainable_variables
regularization_losses
Wlayer_regularization_losses
Xlayer_metrics
Ynon_trainable_variables
	variables
Zmetrics

[layers
\Z
VARIABLE_VALUEconv1d_36/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_36/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
trainable_variables
regularization_losses
\layer_regularization_losses
]layer_metrics
^non_trainable_variables
	variables
_metrics

`layers
\Z
VARIABLE_VALUEconv1d_37/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_37/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1
 

0
 1
?
!trainable_variables
"regularization_losses
alayer_regularization_losses
blayer_metrics
cnon_trainable_variables
#	variables
dmetrics

elayers
\Z
VARIABLE_VALUEconv1d_38/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_38/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
?
'trainable_variables
(regularization_losses
flayer_regularization_losses
glayer_metrics
hnon_trainable_variables
)	variables
imetrics

jlayers
\Z
VARIABLE_VALUEconv1d_39/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_39/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
 

+0
,1
?
-trainable_variables
.regularization_losses
klayer_regularization_losses
llayer_metrics
mnon_trainable_variables
/	variables
nmetrics

olayers
\Z
VARIABLE_VALUEconv1d_40/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_40/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
?
3trainable_variables
4regularization_losses
player_regularization_losses
qlayer_metrics
rnon_trainable_variables
5	variables
smetrics

tlayers
\Z
VARIABLE_VALUEconv1d_41/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_41/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81
 

70
81
?
9trainable_variables
:regularization_losses
ulayer_regularization_losses
vlayer_metrics
wnon_trainable_variables
;	variables
xmetrics

ylayers
 
 
 
?
=trainable_variables
>regularization_losses
zlayer_regularization_losses
{layer_metrics
|non_trainable_variables
?	variables
}metrics

~layers
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1
 

A0
B1
?
Ctrainable_variables
Dregularization_losses
layer_regularization_losses
?layer_metrics
?non_trainable_variables
E	variables
?metrics
?layers
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

G0
H1
 

G0
H1
?
Itrainable_variables
Jregularization_losses
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
K	variables
?metrics
?layers
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
 

?0
?1
V
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
VARIABLE_VALUEAdam/conv1d_35/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_35/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_36/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_36/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_37/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_37/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_38/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_38/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_39/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_39/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_40/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_40/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_41/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_41/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_35/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_35/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_36/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_36/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_37/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_37/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_38/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_38/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_39/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_39/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_40/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_40/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_41/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_41/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
v
serving_default_catPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_convPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_catserving_default_convconv1d_35/kernelconv1d_35/biasconv1d_36/kernelconv1d_36/biasconv1d_37/kernelconv1d_37/biasconv1d_38/kernelconv1d_38/biasconv1d_39/kernelconv1d_39/biasconv1d_40/kernelconv1d_40/biasconv1d_41/kernelconv1d_41/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_700818
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv1d_35/kernel/Read/ReadVariableOp"conv1d_35/bias/Read/ReadVariableOp$conv1d_36/kernel/Read/ReadVariableOp"conv1d_36/bias/Read/ReadVariableOp$conv1d_37/kernel/Read/ReadVariableOp"conv1d_37/bias/Read/ReadVariableOp$conv1d_38/kernel/Read/ReadVariableOp"conv1d_38/bias/Read/ReadVariableOp$conv1d_39/kernel/Read/ReadVariableOp"conv1d_39/bias/Read/ReadVariableOp$conv1d_40/kernel/Read/ReadVariableOp"conv1d_40/bias/Read/ReadVariableOp$conv1d_41/kernel/Read/ReadVariableOp"conv1d_41/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv1d_35/kernel/m/Read/ReadVariableOp)Adam/conv1d_35/bias/m/Read/ReadVariableOp+Adam/conv1d_36/kernel/m/Read/ReadVariableOp)Adam/conv1d_36/bias/m/Read/ReadVariableOp+Adam/conv1d_37/kernel/m/Read/ReadVariableOp)Adam/conv1d_37/bias/m/Read/ReadVariableOp+Adam/conv1d_38/kernel/m/Read/ReadVariableOp)Adam/conv1d_38/bias/m/Read/ReadVariableOp+Adam/conv1d_39/kernel/m/Read/ReadVariableOp)Adam/conv1d_39/bias/m/Read/ReadVariableOp+Adam/conv1d_40/kernel/m/Read/ReadVariableOp)Adam/conv1d_40/bias/m/Read/ReadVariableOp+Adam/conv1d_41/kernel/m/Read/ReadVariableOp)Adam/conv1d_41/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp+Adam/conv1d_35/kernel/v/Read/ReadVariableOp)Adam/conv1d_35/bias/v/Read/ReadVariableOp+Adam/conv1d_36/kernel/v/Read/ReadVariableOp)Adam/conv1d_36/bias/v/Read/ReadVariableOp+Adam/conv1d_37/kernel/v/Read/ReadVariableOp)Adam/conv1d_37/bias/v/Read/ReadVariableOp+Adam/conv1d_38/kernel/v/Read/ReadVariableOp)Adam/conv1d_38/bias/v/Read/ReadVariableOp+Adam/conv1d_39/kernel/v/Read/ReadVariableOp)Adam/conv1d_39/bias/v/Read/ReadVariableOp+Adam/conv1d_40/kernel/v/Read/ReadVariableOp)Adam/conv1d_40/bias/v/Read/ReadVariableOp+Adam/conv1d_41/kernel/v/Read/ReadVariableOp)Adam/conv1d_41/bias/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOpConst*L
TinE
C2A	*
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
__inference__traced_save_701593
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_35/kernelconv1d_35/biasconv1d_36/kernelconv1d_36/biasconv1d_37/kernelconv1d_37/biasconv1d_38/kernelconv1d_38/biasconv1d_39/kernelconv1d_39/biasconv1d_40/kernelconv1d_40/biasconv1d_41/kernelconv1d_41/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv1d_35/kernel/mAdam/conv1d_35/bias/mAdam/conv1d_36/kernel/mAdam/conv1d_36/bias/mAdam/conv1d_37/kernel/mAdam/conv1d_37/bias/mAdam/conv1d_38/kernel/mAdam/conv1d_38/bias/mAdam/conv1d_39/kernel/mAdam/conv1d_39/bias/mAdam/conv1d_40/kernel/mAdam/conv1d_40/bias/mAdam/conv1d_41/kernel/mAdam/conv1d_41/bias/mAdam/dense_10/kernel/mAdam/dense_10/bias/mAdam/dense_11/kernel/mAdam/dense_11/bias/mAdam/conv1d_35/kernel/vAdam/conv1d_35/bias/vAdam/conv1d_36/kernel/vAdam/conv1d_36/bias/vAdam/conv1d_37/kernel/vAdam/conv1d_37/bias/vAdam/conv1d_38/kernel/vAdam/conv1d_38/bias/vAdam/conv1d_39/kernel/vAdam/conv1d_39/bias/vAdam/conv1d_40/kernel/vAdam/conv1d_40/bias/vAdam/conv1d_41/kernel/vAdam/conv1d_41/bias/vAdam/dense_10/kernel/vAdam/dense_10/bias/vAdam/dense_11/kernel/vAdam/dense_11/bias/v*K
TinD
B2@*
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
"__inference__traced_restore_701792ʜ
?
?
E__inference_conv1d_36_layer_call_and_return_conditional_losses_700273

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
?}
?
__inference__traced_save_701593
file_prefix/
+savev2_conv1d_35_kernel_read_readvariableop-
)savev2_conv1d_35_bias_read_readvariableop/
+savev2_conv1d_36_kernel_read_readvariableop-
)savev2_conv1d_36_bias_read_readvariableop/
+savev2_conv1d_37_kernel_read_readvariableop-
)savev2_conv1d_37_bias_read_readvariableop/
+savev2_conv1d_38_kernel_read_readvariableop-
)savev2_conv1d_38_bias_read_readvariableop/
+savev2_conv1d_39_kernel_read_readvariableop-
)savev2_conv1d_39_bias_read_readvariableop/
+savev2_conv1d_40_kernel_read_readvariableop-
)savev2_conv1d_40_bias_read_readvariableop/
+savev2_conv1d_41_kernel_read_readvariableop-
)savev2_conv1d_41_bias_read_readvariableop.
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
2savev2_adam_conv1d_35_kernel_m_read_readvariableop4
0savev2_adam_conv1d_35_bias_m_read_readvariableop6
2savev2_adam_conv1d_36_kernel_m_read_readvariableop4
0savev2_adam_conv1d_36_bias_m_read_readvariableop6
2savev2_adam_conv1d_37_kernel_m_read_readvariableop4
0savev2_adam_conv1d_37_bias_m_read_readvariableop6
2savev2_adam_conv1d_38_kernel_m_read_readvariableop4
0savev2_adam_conv1d_38_bias_m_read_readvariableop6
2savev2_adam_conv1d_39_kernel_m_read_readvariableop4
0savev2_adam_conv1d_39_bias_m_read_readvariableop6
2savev2_adam_conv1d_40_kernel_m_read_readvariableop4
0savev2_adam_conv1d_40_bias_m_read_readvariableop6
2savev2_adam_conv1d_41_kernel_m_read_readvariableop4
0savev2_adam_conv1d_41_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableop6
2savev2_adam_conv1d_35_kernel_v_read_readvariableop4
0savev2_adam_conv1d_35_bias_v_read_readvariableop6
2savev2_adam_conv1d_36_kernel_v_read_readvariableop4
0savev2_adam_conv1d_36_bias_v_read_readvariableop6
2savev2_adam_conv1d_37_kernel_v_read_readvariableop4
0savev2_adam_conv1d_37_bias_v_read_readvariableop6
2savev2_adam_conv1d_38_kernel_v_read_readvariableop4
0savev2_adam_conv1d_38_bias_v_read_readvariableop6
2savev2_adam_conv1d_39_kernel_v_read_readvariableop4
0savev2_adam_conv1d_39_bias_v_read_readvariableop6
2savev2_adam_conv1d_40_kernel_v_read_readvariableop4
0savev2_adam_conv1d_40_bias_v_read_readvariableop6
2savev2_adam_conv1d_41_kernel_v_read_readvariableop4
0savev2_adam_conv1d_41_bias_v_read_readvariableop5
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
ShardedFilename?#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?"
value?"B?"@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?
value?B?@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv1d_35_kernel_read_readvariableop)savev2_conv1d_35_bias_read_readvariableop+savev2_conv1d_36_kernel_read_readvariableop)savev2_conv1d_36_bias_read_readvariableop+savev2_conv1d_37_kernel_read_readvariableop)savev2_conv1d_37_bias_read_readvariableop+savev2_conv1d_38_kernel_read_readvariableop)savev2_conv1d_38_bias_read_readvariableop+savev2_conv1d_39_kernel_read_readvariableop)savev2_conv1d_39_bias_read_readvariableop+savev2_conv1d_40_kernel_read_readvariableop)savev2_conv1d_40_bias_read_readvariableop+savev2_conv1d_41_kernel_read_readvariableop)savev2_conv1d_41_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv1d_35_kernel_m_read_readvariableop0savev2_adam_conv1d_35_bias_m_read_readvariableop2savev2_adam_conv1d_36_kernel_m_read_readvariableop0savev2_adam_conv1d_36_bias_m_read_readvariableop2savev2_adam_conv1d_37_kernel_m_read_readvariableop0savev2_adam_conv1d_37_bias_m_read_readvariableop2savev2_adam_conv1d_38_kernel_m_read_readvariableop0savev2_adam_conv1d_38_bias_m_read_readvariableop2savev2_adam_conv1d_39_kernel_m_read_readvariableop0savev2_adam_conv1d_39_bias_m_read_readvariableop2savev2_adam_conv1d_40_kernel_m_read_readvariableop0savev2_adam_conv1d_40_bias_m_read_readvariableop2savev2_adam_conv1d_41_kernel_m_read_readvariableop0savev2_adam_conv1d_41_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableop2savev2_adam_conv1d_35_kernel_v_read_readvariableop0savev2_adam_conv1d_35_bias_v_read_readvariableop2savev2_adam_conv1d_36_kernel_v_read_readvariableop0savev2_adam_conv1d_36_bias_v_read_readvariableop2savev2_adam_conv1d_37_kernel_v_read_readvariableop0savev2_adam_conv1d_37_bias_v_read_readvariableop2savev2_adam_conv1d_38_kernel_v_read_readvariableop0savev2_adam_conv1d_38_bias_v_read_readvariableop2savev2_adam_conv1d_39_kernel_v_read_readvariableop0savev2_adam_conv1d_39_bias_v_read_readvariableop2savev2_adam_conv1d_40_kernel_v_read_readvariableop0savev2_adam_conv1d_40_bias_v_read_readvariableop2savev2_adam_conv1d_41_kernel_v_read_readvariableop0savev2_adam_conv1d_41_bias_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@	2
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
?: :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:	?:::: : : : : : : : : :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:	?::::@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:	?:::: 2(
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
:@:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :
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
: :($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 
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
:@:%*!

_output_shapes
:	?: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::(.$
"
_output_shapes
:@: /

_output_shapes
:@:(0$
"
_output_shapes
:@@: 1

_output_shapes
:@:(2$
"
_output_shapes
:@@: 3
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
:@:%<!

_output_shapes
:	?: =

_output_shapes
::$> 

_output_shapes

:: ?

_output_shapes
::@

_output_shapes
: 
?
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_701335

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
?

?
D__inference_dense_11_layer_call_and_return_conditional_losses_701371

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
)__inference_dense_10_layer_call_fn_701360

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_7004842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?3
?
C__inference_model_5_layer_call_and_return_conditional_losses_700634

inputs
inputs_1
conv1d_35_700587
conv1d_35_700589
conv1d_36_700592
conv1d_36_700594
conv1d_37_700597
conv1d_37_700599
conv1d_38_700602
conv1d_38_700604
conv1d_39_700607
conv1d_39_700609
conv1d_40_700612
conv1d_40_700614
conv1d_41_700617
conv1d_41_700619
dense_10_700623
dense_10_700625
dense_11_700628
dense_11_700630
identity??!conv1d_35/StatefulPartitionedCall?!conv1d_36/StatefulPartitionedCall?!conv1d_37/StatefulPartitionedCall?!conv1d_38/StatefulPartitionedCall?!conv1d_39/StatefulPartitionedCall?!conv1d_40/StatefulPartitionedCall?!conv1d_41/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
!conv1d_35/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_35_700587conv1d_35_700589*
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
E__inference_conv1d_35_layer_call_and_return_conditional_losses_7002392#
!conv1d_35/StatefulPartitionedCall?
!conv1d_36/StatefulPartitionedCallStatefulPartitionedCall*conv1d_35/StatefulPartitionedCall:output:0conv1d_36_700592conv1d_36_700594*
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
E__inference_conv1d_36_layer_call_and_return_conditional_losses_7002732#
!conv1d_36/StatefulPartitionedCall?
!conv1d_37/StatefulPartitionedCallStatefulPartitionedCall*conv1d_36/StatefulPartitionedCall:output:0conv1d_37_700597conv1d_37_700599*
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
E__inference_conv1d_37_layer_call_and_return_conditional_losses_7003072#
!conv1d_37/StatefulPartitionedCall?
!conv1d_38/StatefulPartitionedCallStatefulPartitionedCall*conv1d_37/StatefulPartitionedCall:output:0conv1d_38_700602conv1d_38_700604*
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
E__inference_conv1d_38_layer_call_and_return_conditional_losses_7003412#
!conv1d_38/StatefulPartitionedCall?
!conv1d_39/StatefulPartitionedCallStatefulPartitionedCall*conv1d_38/StatefulPartitionedCall:output:0conv1d_39_700607conv1d_39_700609*
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
E__inference_conv1d_39_layer_call_and_return_conditional_losses_7003752#
!conv1d_39/StatefulPartitionedCall?
!conv1d_40/StatefulPartitionedCallStatefulPartitionedCall*conv1d_39/StatefulPartitionedCall:output:0conv1d_40_700612conv1d_40_700614*
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
E__inference_conv1d_40_layer_call_and_return_conditional_losses_7004092#
!conv1d_40/StatefulPartitionedCall?
!conv1d_41/StatefulPartitionedCallStatefulPartitionedCall*conv1d_40/StatefulPartitionedCall:output:0conv1d_41_700617conv1d_41_700619*
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
E__inference_conv1d_41_layer_call_and_return_conditional_losses_7004432#
!conv1d_41/StatefulPartitionedCall?
flatten_5/PartitionedCallPartitionedCall*conv1d_41/StatefulPartitionedCall:output:0*
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_7004652
flatten_5/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_10_700623dense_10_700625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_7004842"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_700628dense_11_700630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_7005112"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0"^conv1d_35/StatefulPartitionedCall"^conv1d_36/StatefulPartitionedCall"^conv1d_37/StatefulPartitionedCall"^conv1d_38/StatefulPartitionedCall"^conv1d_39/StatefulPartitionedCall"^conv1d_40/StatefulPartitionedCall"^conv1d_41/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2F
!conv1d_35/StatefulPartitionedCall!conv1d_35/StatefulPartitionedCall2F
!conv1d_36/StatefulPartitionedCall!conv1d_36/StatefulPartitionedCall2F
!conv1d_37/StatefulPartitionedCall!conv1d_37/StatefulPartitionedCall2F
!conv1d_38/StatefulPartitionedCall!conv1d_38/StatefulPartitionedCall2F
!conv1d_39/StatefulPartitionedCall!conv1d_39/StatefulPartitionedCall2F
!conv1d_40/StatefulPartitionedCall!conv1d_40/StatefulPartitionedCall2F
!conv1d_41/StatefulPartitionedCall!conv1d_41/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv1d_35_layer_call_and_return_conditional_losses_700239

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
?
?
E__inference_conv1d_38_layer_call_and_return_conditional_losses_700341

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
??
?
C__inference_model_5_layer_call_and_return_conditional_losses_701056
inputs_0
inputs_19
5conv1d_35_conv1d_expanddims_1_readvariableop_resource-
)conv1d_35_biasadd_readvariableop_resource9
5conv1d_36_conv1d_expanddims_1_readvariableop_resource-
)conv1d_36_biasadd_readvariableop_resource9
5conv1d_37_conv1d_expanddims_1_readvariableop_resource-
)conv1d_37_biasadd_readvariableop_resource9
5conv1d_38_conv1d_expanddims_1_readvariableop_resource-
)conv1d_38_biasadd_readvariableop_resource9
5conv1d_39_conv1d_expanddims_1_readvariableop_resource-
)conv1d_39_biasadd_readvariableop_resource9
5conv1d_40_conv1d_expanddims_1_readvariableop_resource-
)conv1d_40_biasadd_readvariableop_resource9
5conv1d_41_conv1d_expanddims_1_readvariableop_resource-
)conv1d_41_biasadd_readvariableop_resource.
*dense_10_mlcmatmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource.
*dense_11_mlcmatmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity?? conv1d_35/BiasAdd/ReadVariableOp?,conv1d_35/conv1d/ExpandDims_1/ReadVariableOp? conv1d_36/BiasAdd/ReadVariableOp?,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp? conv1d_37/BiasAdd/ReadVariableOp?,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp? conv1d_38/BiasAdd/ReadVariableOp?,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp? conv1d_39/BiasAdd/ReadVariableOp?,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp? conv1d_40/BiasAdd/ReadVariableOp?,conv1d_40/conv1d/ExpandDims_1/ReadVariableOp? conv1d_41/BiasAdd/ReadVariableOp?,conv1d_41/conv1d/ExpandDims_1/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?!dense_10/MLCMatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?!dense_11/MLCMatMul/ReadVariableOp?
conv1d_35/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_35/Pad/paddings?
conv1d_35/PadPadinputs_0conv1d_35/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
conv1d_35/Pad?
conv1d_35/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_35/conv1d/ExpandDims/dim?
conv1d_35/conv1d/ExpandDims
ExpandDimsconv1d_35/Pad:output:0(conv1d_35/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_35/conv1d/ExpandDims?
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_35_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_35/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_35/conv1d/ExpandDims_1/dim?
conv1d_35/conv1d/ExpandDims_1
ExpandDims4conv1d_35/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_35/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_35/conv1d/ExpandDims_1?
conv1d_35/conv1d	MLCConv2D$conv1d_35/conv1d/ExpandDims:output:0&conv1d_35/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_35/conv1d?
conv1d_35/conv1d/SqueezeSqueezeconv1d_35/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_35/conv1d/Squeeze?
 conv1d_35/BiasAdd/ReadVariableOpReadVariableOp)conv1d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_35/BiasAdd/ReadVariableOp?
conv1d_35/BiasAddBiasAdd!conv1d_35/conv1d/Squeeze:output:0(conv1d_35/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_35/BiasAddz
conv1d_35/ReluReluconv1d_35/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_35/Relu?
conv1d_36/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_36/Pad/paddings?
conv1d_36/PadPadconv1d_35/Relu:activations:0conv1d_36/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_36/Pad?
conv1d_36/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_36/conv1d/ExpandDims/dim?
conv1d_36/conv1d/ExpandDims
ExpandDimsconv1d_36/Pad:output:0(conv1d_36/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_36/conv1d/ExpandDims?
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_36_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_36/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_36/conv1d/ExpandDims_1/dim?
conv1d_36/conv1d/ExpandDims_1
ExpandDims4conv1d_36/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_36/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_36/conv1d/ExpandDims_1?
conv1d_36/conv1d	MLCConv2D$conv1d_36/conv1d/ExpandDims:output:0&conv1d_36/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_36/conv1d?
conv1d_36/conv1d/SqueezeSqueezeconv1d_36/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_36/conv1d/Squeeze?
 conv1d_36/BiasAdd/ReadVariableOpReadVariableOp)conv1d_36_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_36/BiasAdd/ReadVariableOp?
conv1d_36/BiasAddBiasAdd!conv1d_36/conv1d/Squeeze:output:0(conv1d_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_36/BiasAddz
conv1d_36/ReluReluconv1d_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_36/Relu?
conv1d_37/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_37/Pad/paddings?
conv1d_37/PadPadconv1d_36/Relu:activations:0conv1d_37/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_37/Pad?
conv1d_37/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_37/conv1d/ExpandDims/dim?
conv1d_37/conv1d/ExpandDims
ExpandDimsconv1d_37/Pad:output:0(conv1d_37/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_37/conv1d/ExpandDims?
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_37/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_37/conv1d/ExpandDims_1/dim?
conv1d_37/conv1d/ExpandDims_1
ExpandDims4conv1d_37/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_37/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_37/conv1d/ExpandDims_1?
conv1d_37/conv1d	MLCConv2D$conv1d_37/conv1d/ExpandDims:output:0&conv1d_37/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_37/conv1d?
conv1d_37/conv1d/SqueezeSqueezeconv1d_37/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_37/conv1d/Squeeze?
 conv1d_37/BiasAdd/ReadVariableOpReadVariableOp)conv1d_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_37/BiasAdd/ReadVariableOp?
conv1d_37/BiasAddBiasAdd!conv1d_37/conv1d/Squeeze:output:0(conv1d_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_37/BiasAddz
conv1d_37/ReluReluconv1d_37/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_37/Relu?
conv1d_38/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_38/Pad/paddings?
conv1d_38/PadPadconv1d_37/Relu:activations:0conv1d_38/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_38/Pad?
conv1d_38/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_38/conv1d/ExpandDims/dim?
conv1d_38/conv1d/ExpandDims
ExpandDimsconv1d_38/Pad:output:0(conv1d_38/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_38/conv1d/ExpandDims?
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_38/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_38/conv1d/ExpandDims_1/dim?
conv1d_38/conv1d/ExpandDims_1
ExpandDims4conv1d_38/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_38/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_38/conv1d/ExpandDims_1?
conv1d_38/conv1d	MLCConv2D$conv1d_38/conv1d/ExpandDims:output:0&conv1d_38/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_38/conv1d?
conv1d_38/conv1d/SqueezeSqueezeconv1d_38/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_38/conv1d/Squeeze?
 conv1d_38/BiasAdd/ReadVariableOpReadVariableOp)conv1d_38_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_38/BiasAdd/ReadVariableOp?
conv1d_38/BiasAddBiasAdd!conv1d_38/conv1d/Squeeze:output:0(conv1d_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_38/BiasAddz
conv1d_38/ReluReluconv1d_38/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_38/Relu?
conv1d_39/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_39/Pad/paddings?
conv1d_39/PadPadconv1d_38/Relu:activations:0conv1d_39/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_39/Pad?
conv1d_39/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_39/conv1d/ExpandDims/dim?
conv1d_39/conv1d/ExpandDims
ExpandDimsconv1d_39/Pad:output:0(conv1d_39/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_39/conv1d/ExpandDims?
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_39_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_39/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_39/conv1d/ExpandDims_1/dim?
conv1d_39/conv1d/ExpandDims_1
ExpandDims4conv1d_39/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_39/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_39/conv1d/ExpandDims_1?
conv1d_39/conv1d	MLCConv2D$conv1d_39/conv1d/ExpandDims:output:0&conv1d_39/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_39/conv1d?
conv1d_39/conv1d/SqueezeSqueezeconv1d_39/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_39/conv1d/Squeeze?
 conv1d_39/BiasAdd/ReadVariableOpReadVariableOp)conv1d_39_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_39/BiasAdd/ReadVariableOp?
conv1d_39/BiasAddBiasAdd!conv1d_39/conv1d/Squeeze:output:0(conv1d_39/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_39/BiasAddz
conv1d_39/ReluReluconv1d_39/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_39/Relu?
conv1d_40/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_40/Pad/paddings?
conv1d_40/PadPadconv1d_39/Relu:activations:0conv1d_40/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_40/Pad?
conv1d_40/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_40/conv1d/ExpandDims/dim?
conv1d_40/conv1d/ExpandDims
ExpandDimsconv1d_40/Pad:output:0(conv1d_40/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_40/conv1d/ExpandDims?
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_40/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_40/conv1d/ExpandDims_1/dim?
conv1d_40/conv1d/ExpandDims_1
ExpandDims4conv1d_40/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_40/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_40/conv1d/ExpandDims_1?
conv1d_40/conv1d	MLCConv2D$conv1d_40/conv1d/ExpandDims:output:0&conv1d_40/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_40/conv1d?
conv1d_40/conv1d/SqueezeSqueezeconv1d_40/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_40/conv1d/Squeeze?
 conv1d_40/BiasAdd/ReadVariableOpReadVariableOp)conv1d_40_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_40/BiasAdd/ReadVariableOp?
conv1d_40/BiasAddBiasAdd!conv1d_40/conv1d/Squeeze:output:0(conv1d_40/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_40/BiasAddz
conv1d_40/ReluReluconv1d_40/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_40/Relu?
conv1d_41/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_41/Pad/paddings?
conv1d_41/PadPadconv1d_40/Relu:activations:0conv1d_41/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_41/Pad?
conv1d_41/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_41/conv1d/ExpandDims/dim?
conv1d_41/conv1d/ExpandDims
ExpandDimsconv1d_41/Pad:output:0(conv1d_41/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_41/conv1d/ExpandDims?
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_41/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_41/conv1d/ExpandDims_1/dim?
conv1d_41/conv1d/ExpandDims_1
ExpandDims4conv1d_41/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_41/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_41/conv1d/ExpandDims_1?
conv1d_41/conv1d	MLCConv2D$conv1d_41/conv1d/ExpandDims:output:0&conv1d_41/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_41/conv1d?
conv1d_41/conv1d/SqueezeSqueezeconv1d_41/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_41/conv1d/Squeeze?
 conv1d_41/BiasAdd/ReadVariableOpReadVariableOp)conv1d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_41/BiasAdd/ReadVariableOp?
conv1d_41/BiasAddBiasAdd!conv1d_41/conv1d/Squeeze:output:0(conv1d_41/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_41/BiasAddz
conv1d_41/ReluReluconv1d_41/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_41/Relus
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_5/Const?
flatten_5/ReshapeReshapeconv1d_41/Relu:activations:0flatten_5/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_5/Reshape?
!dense_10/MLCMatMul/ReadVariableOpReadVariableOp*dense_10_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_10/MLCMatMul/ReadVariableOp?
dense_10/MLCMatMul	MLCMatMulflatten_5/Reshape:output:0)dense_10/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/MLCMatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MLCMatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_10/Relu?
!dense_11/MLCMatMul/ReadVariableOpReadVariableOp*dense_11_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_11/MLCMatMul/ReadVariableOp?
dense_11/MLCMatMul	MLCMatMuldense_10/Relu:activations:0)dense_11/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/MLCMatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MLCMatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_11/Softmax?
IdentityIdentitydense_11/Softmax:softmax:0!^conv1d_35/BiasAdd/ReadVariableOp-^conv1d_35/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_36/BiasAdd/ReadVariableOp-^conv1d_36/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_37/BiasAdd/ReadVariableOp-^conv1d_37/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_38/BiasAdd/ReadVariableOp-^conv1d_38/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_39/BiasAdd/ReadVariableOp-^conv1d_39/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_40/BiasAdd/ReadVariableOp-^conv1d_40/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_41/BiasAdd/ReadVariableOp-^conv1d_41/conv1d/ExpandDims_1/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/MLCMatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2D
 conv1d_35/BiasAdd/ReadVariableOp conv1d_35/BiasAdd/ReadVariableOp2\
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOp,conv1d_35/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_36/BiasAdd/ReadVariableOp conv1d_36/BiasAdd/ReadVariableOp2\
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_37/BiasAdd/ReadVariableOp conv1d_37/BiasAdd/ReadVariableOp2\
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_38/BiasAdd/ReadVariableOp conv1d_38/BiasAdd/ReadVariableOp2\
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_39/BiasAdd/ReadVariableOp conv1d_39/BiasAdd/ReadVariableOp2\
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_40/BiasAdd/ReadVariableOp conv1d_40/BiasAdd/ReadVariableOp2\
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOp,conv1d_40/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_41/BiasAdd/ReadVariableOp conv1d_41/BiasAdd/ReadVariableOp2\
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOp,conv1d_41/conv1d/ExpandDims_1/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/MLCMatMul/ReadVariableOp!dense_10/MLCMatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/MLCMatMul/ReadVariableOp!dense_11/MLCMatMul/ReadVariableOp:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?3
?
C__inference_model_5_layer_call_and_return_conditional_losses_700727

inputs
inputs_1
conv1d_35_700680
conv1d_35_700682
conv1d_36_700685
conv1d_36_700687
conv1d_37_700690
conv1d_37_700692
conv1d_38_700695
conv1d_38_700697
conv1d_39_700700
conv1d_39_700702
conv1d_40_700705
conv1d_40_700707
conv1d_41_700710
conv1d_41_700712
dense_10_700716
dense_10_700718
dense_11_700721
dense_11_700723
identity??!conv1d_35/StatefulPartitionedCall?!conv1d_36/StatefulPartitionedCall?!conv1d_37/StatefulPartitionedCall?!conv1d_38/StatefulPartitionedCall?!conv1d_39/StatefulPartitionedCall?!conv1d_40/StatefulPartitionedCall?!conv1d_41/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
!conv1d_35/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_35_700680conv1d_35_700682*
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
E__inference_conv1d_35_layer_call_and_return_conditional_losses_7002392#
!conv1d_35/StatefulPartitionedCall?
!conv1d_36/StatefulPartitionedCallStatefulPartitionedCall*conv1d_35/StatefulPartitionedCall:output:0conv1d_36_700685conv1d_36_700687*
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
E__inference_conv1d_36_layer_call_and_return_conditional_losses_7002732#
!conv1d_36/StatefulPartitionedCall?
!conv1d_37/StatefulPartitionedCallStatefulPartitionedCall*conv1d_36/StatefulPartitionedCall:output:0conv1d_37_700690conv1d_37_700692*
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
E__inference_conv1d_37_layer_call_and_return_conditional_losses_7003072#
!conv1d_37/StatefulPartitionedCall?
!conv1d_38/StatefulPartitionedCallStatefulPartitionedCall*conv1d_37/StatefulPartitionedCall:output:0conv1d_38_700695conv1d_38_700697*
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
E__inference_conv1d_38_layer_call_and_return_conditional_losses_7003412#
!conv1d_38/StatefulPartitionedCall?
!conv1d_39/StatefulPartitionedCallStatefulPartitionedCall*conv1d_38/StatefulPartitionedCall:output:0conv1d_39_700700conv1d_39_700702*
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
E__inference_conv1d_39_layer_call_and_return_conditional_losses_7003752#
!conv1d_39/StatefulPartitionedCall?
!conv1d_40/StatefulPartitionedCallStatefulPartitionedCall*conv1d_39/StatefulPartitionedCall:output:0conv1d_40_700705conv1d_40_700707*
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
E__inference_conv1d_40_layer_call_and_return_conditional_losses_7004092#
!conv1d_40/StatefulPartitionedCall?
!conv1d_41/StatefulPartitionedCallStatefulPartitionedCall*conv1d_40/StatefulPartitionedCall:output:0conv1d_41_700710conv1d_41_700712*
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
E__inference_conv1d_41_layer_call_and_return_conditional_losses_7004432#
!conv1d_41/StatefulPartitionedCall?
flatten_5/PartitionedCallPartitionedCall*conv1d_41/StatefulPartitionedCall:output:0*
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_7004652
flatten_5/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_10_700716dense_10_700718*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_7004842"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_700721dense_11_700723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_7005112"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0"^conv1d_35/StatefulPartitionedCall"^conv1d_36/StatefulPartitionedCall"^conv1d_37/StatefulPartitionedCall"^conv1d_38/StatefulPartitionedCall"^conv1d_39/StatefulPartitionedCall"^conv1d_40/StatefulPartitionedCall"^conv1d_41/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2F
!conv1d_35/StatefulPartitionedCall!conv1d_35/StatefulPartitionedCall2F
!conv1d_36/StatefulPartitionedCall!conv1d_36/StatefulPartitionedCall2F
!conv1d_37/StatefulPartitionedCall!conv1d_37/StatefulPartitionedCall2F
!conv1d_38/StatefulPartitionedCall!conv1d_38/StatefulPartitionedCall2F
!conv1d_39/StatefulPartitionedCall!conv1d_39/StatefulPartitionedCall2F
!conv1d_40/StatefulPartitionedCall!conv1d_40/StatefulPartitionedCall2F
!conv1d_41/StatefulPartitionedCall!conv1d_41/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_conv1d_37_layer_call_fn_701221

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
E__inference_conv1d_37_layer_call_and_return_conditional_losses_7003072
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
~
)__inference_dense_11_layer_call_fn_701380

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_7005112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_model_5_layer_call_fn_701098
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

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_7006342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?3
?
C__inference_model_5_layer_call_and_return_conditional_losses_700528
conv
cat
conv1d_35_700250
conv1d_35_700252
conv1d_36_700284
conv1d_36_700286
conv1d_37_700318
conv1d_37_700320
conv1d_38_700352
conv1d_38_700354
conv1d_39_700386
conv1d_39_700388
conv1d_40_700420
conv1d_40_700422
conv1d_41_700454
conv1d_41_700456
dense_10_700495
dense_10_700497
dense_11_700522
dense_11_700524
identity??!conv1d_35/StatefulPartitionedCall?!conv1d_36/StatefulPartitionedCall?!conv1d_37/StatefulPartitionedCall?!conv1d_38/StatefulPartitionedCall?!conv1d_39/StatefulPartitionedCall?!conv1d_40/StatefulPartitionedCall?!conv1d_41/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
!conv1d_35/StatefulPartitionedCallStatefulPartitionedCallconvconv1d_35_700250conv1d_35_700252*
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
E__inference_conv1d_35_layer_call_and_return_conditional_losses_7002392#
!conv1d_35/StatefulPartitionedCall?
!conv1d_36/StatefulPartitionedCallStatefulPartitionedCall*conv1d_35/StatefulPartitionedCall:output:0conv1d_36_700284conv1d_36_700286*
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
E__inference_conv1d_36_layer_call_and_return_conditional_losses_7002732#
!conv1d_36/StatefulPartitionedCall?
!conv1d_37/StatefulPartitionedCallStatefulPartitionedCall*conv1d_36/StatefulPartitionedCall:output:0conv1d_37_700318conv1d_37_700320*
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
E__inference_conv1d_37_layer_call_and_return_conditional_losses_7003072#
!conv1d_37/StatefulPartitionedCall?
!conv1d_38/StatefulPartitionedCallStatefulPartitionedCall*conv1d_37/StatefulPartitionedCall:output:0conv1d_38_700352conv1d_38_700354*
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
E__inference_conv1d_38_layer_call_and_return_conditional_losses_7003412#
!conv1d_38/StatefulPartitionedCall?
!conv1d_39/StatefulPartitionedCallStatefulPartitionedCall*conv1d_38/StatefulPartitionedCall:output:0conv1d_39_700386conv1d_39_700388*
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
E__inference_conv1d_39_layer_call_and_return_conditional_losses_7003752#
!conv1d_39/StatefulPartitionedCall?
!conv1d_40/StatefulPartitionedCallStatefulPartitionedCall*conv1d_39/StatefulPartitionedCall:output:0conv1d_40_700420conv1d_40_700422*
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
E__inference_conv1d_40_layer_call_and_return_conditional_losses_7004092#
!conv1d_40/StatefulPartitionedCall?
!conv1d_41/StatefulPartitionedCallStatefulPartitionedCall*conv1d_40/StatefulPartitionedCall:output:0conv1d_41_700454conv1d_41_700456*
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
E__inference_conv1d_41_layer_call_and_return_conditional_losses_7004432#
!conv1d_41/StatefulPartitionedCall?
flatten_5/PartitionedCallPartitionedCall*conv1d_41/StatefulPartitionedCall:output:0*
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_7004652
flatten_5/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_10_700495dense_10_700497*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_7004842"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_700522dense_11_700524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_7005112"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0"^conv1d_35/StatefulPartitionedCall"^conv1d_36/StatefulPartitionedCall"^conv1d_37/StatefulPartitionedCall"^conv1d_38/StatefulPartitionedCall"^conv1d_39/StatefulPartitionedCall"^conv1d_40/StatefulPartitionedCall"^conv1d_41/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2F
!conv1d_35/StatefulPartitionedCall!conv1d_35/StatefulPartitionedCall2F
!conv1d_36/StatefulPartitionedCall!conv1d_36/StatefulPartitionedCall2F
!conv1d_37/StatefulPartitionedCall!conv1d_37/StatefulPartitionedCall2F
!conv1d_38/StatefulPartitionedCall!conv1d_38/StatefulPartitionedCall2F
!conv1d_39/StatefulPartitionedCall!conv1d_39/StatefulPartitionedCall2F
!conv1d_40/StatefulPartitionedCall!conv1d_40/StatefulPartitionedCall2F
!conv1d_41/StatefulPartitionedCall!conv1d_41/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
?
?
(__inference_model_5_layer_call_fn_700673
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

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconvcatunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_7006342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
?

*__inference_conv1d_41_layer_call_fn_701329

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
E__inference_conv1d_41_layer_call_and_return_conditional_losses_7004432
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
?

?
D__inference_dense_11_layer_call_and_return_conditional_losses_700511

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv1d_39_layer_call_and_return_conditional_losses_701266

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
??
?
C__inference_model_5_layer_call_and_return_conditional_losses_700937
inputs_0
inputs_19
5conv1d_35_conv1d_expanddims_1_readvariableop_resource-
)conv1d_35_biasadd_readvariableop_resource9
5conv1d_36_conv1d_expanddims_1_readvariableop_resource-
)conv1d_36_biasadd_readvariableop_resource9
5conv1d_37_conv1d_expanddims_1_readvariableop_resource-
)conv1d_37_biasadd_readvariableop_resource9
5conv1d_38_conv1d_expanddims_1_readvariableop_resource-
)conv1d_38_biasadd_readvariableop_resource9
5conv1d_39_conv1d_expanddims_1_readvariableop_resource-
)conv1d_39_biasadd_readvariableop_resource9
5conv1d_40_conv1d_expanddims_1_readvariableop_resource-
)conv1d_40_biasadd_readvariableop_resource9
5conv1d_41_conv1d_expanddims_1_readvariableop_resource-
)conv1d_41_biasadd_readvariableop_resource.
*dense_10_mlcmatmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource.
*dense_11_mlcmatmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity?? conv1d_35/BiasAdd/ReadVariableOp?,conv1d_35/conv1d/ExpandDims_1/ReadVariableOp? conv1d_36/BiasAdd/ReadVariableOp?,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp? conv1d_37/BiasAdd/ReadVariableOp?,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp? conv1d_38/BiasAdd/ReadVariableOp?,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp? conv1d_39/BiasAdd/ReadVariableOp?,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp? conv1d_40/BiasAdd/ReadVariableOp?,conv1d_40/conv1d/ExpandDims_1/ReadVariableOp? conv1d_41/BiasAdd/ReadVariableOp?,conv1d_41/conv1d/ExpandDims_1/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?!dense_10/MLCMatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?!dense_11/MLCMatMul/ReadVariableOp?
conv1d_35/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_35/Pad/paddings?
conv1d_35/PadPadinputs_0conv1d_35/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
conv1d_35/Pad?
conv1d_35/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_35/conv1d/ExpandDims/dim?
conv1d_35/conv1d/ExpandDims
ExpandDimsconv1d_35/Pad:output:0(conv1d_35/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_35/conv1d/ExpandDims?
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_35_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_35/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_35/conv1d/ExpandDims_1/dim?
conv1d_35/conv1d/ExpandDims_1
ExpandDims4conv1d_35/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_35/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_35/conv1d/ExpandDims_1?
conv1d_35/conv1d	MLCConv2D$conv1d_35/conv1d/ExpandDims:output:0&conv1d_35/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_35/conv1d?
conv1d_35/conv1d/SqueezeSqueezeconv1d_35/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_35/conv1d/Squeeze?
 conv1d_35/BiasAdd/ReadVariableOpReadVariableOp)conv1d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_35/BiasAdd/ReadVariableOp?
conv1d_35/BiasAddBiasAdd!conv1d_35/conv1d/Squeeze:output:0(conv1d_35/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_35/BiasAddz
conv1d_35/ReluReluconv1d_35/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_35/Relu?
conv1d_36/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_36/Pad/paddings?
conv1d_36/PadPadconv1d_35/Relu:activations:0conv1d_36/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_36/Pad?
conv1d_36/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_36/conv1d/ExpandDims/dim?
conv1d_36/conv1d/ExpandDims
ExpandDimsconv1d_36/Pad:output:0(conv1d_36/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_36/conv1d/ExpandDims?
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_36_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_36/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_36/conv1d/ExpandDims_1/dim?
conv1d_36/conv1d/ExpandDims_1
ExpandDims4conv1d_36/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_36/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_36/conv1d/ExpandDims_1?
conv1d_36/conv1d	MLCConv2D$conv1d_36/conv1d/ExpandDims:output:0&conv1d_36/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_36/conv1d?
conv1d_36/conv1d/SqueezeSqueezeconv1d_36/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_36/conv1d/Squeeze?
 conv1d_36/BiasAdd/ReadVariableOpReadVariableOp)conv1d_36_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_36/BiasAdd/ReadVariableOp?
conv1d_36/BiasAddBiasAdd!conv1d_36/conv1d/Squeeze:output:0(conv1d_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_36/BiasAddz
conv1d_36/ReluReluconv1d_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_36/Relu?
conv1d_37/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_37/Pad/paddings?
conv1d_37/PadPadconv1d_36/Relu:activations:0conv1d_37/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_37/Pad?
conv1d_37/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_37/conv1d/ExpandDims/dim?
conv1d_37/conv1d/ExpandDims
ExpandDimsconv1d_37/Pad:output:0(conv1d_37/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_37/conv1d/ExpandDims?
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_37/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_37/conv1d/ExpandDims_1/dim?
conv1d_37/conv1d/ExpandDims_1
ExpandDims4conv1d_37/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_37/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_37/conv1d/ExpandDims_1?
conv1d_37/conv1d	MLCConv2D$conv1d_37/conv1d/ExpandDims:output:0&conv1d_37/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_37/conv1d?
conv1d_37/conv1d/SqueezeSqueezeconv1d_37/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_37/conv1d/Squeeze?
 conv1d_37/BiasAdd/ReadVariableOpReadVariableOp)conv1d_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_37/BiasAdd/ReadVariableOp?
conv1d_37/BiasAddBiasAdd!conv1d_37/conv1d/Squeeze:output:0(conv1d_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_37/BiasAddz
conv1d_37/ReluReluconv1d_37/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_37/Relu?
conv1d_38/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_38/Pad/paddings?
conv1d_38/PadPadconv1d_37/Relu:activations:0conv1d_38/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_38/Pad?
conv1d_38/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_38/conv1d/ExpandDims/dim?
conv1d_38/conv1d/ExpandDims
ExpandDimsconv1d_38/Pad:output:0(conv1d_38/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_38/conv1d/ExpandDims?
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_38/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_38/conv1d/ExpandDims_1/dim?
conv1d_38/conv1d/ExpandDims_1
ExpandDims4conv1d_38/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_38/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_38/conv1d/ExpandDims_1?
conv1d_38/conv1d	MLCConv2D$conv1d_38/conv1d/ExpandDims:output:0&conv1d_38/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_38/conv1d?
conv1d_38/conv1d/SqueezeSqueezeconv1d_38/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_38/conv1d/Squeeze?
 conv1d_38/BiasAdd/ReadVariableOpReadVariableOp)conv1d_38_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_38/BiasAdd/ReadVariableOp?
conv1d_38/BiasAddBiasAdd!conv1d_38/conv1d/Squeeze:output:0(conv1d_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_38/BiasAddz
conv1d_38/ReluReluconv1d_38/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_38/Relu?
conv1d_39/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_39/Pad/paddings?
conv1d_39/PadPadconv1d_38/Relu:activations:0conv1d_39/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_39/Pad?
conv1d_39/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_39/conv1d/ExpandDims/dim?
conv1d_39/conv1d/ExpandDims
ExpandDimsconv1d_39/Pad:output:0(conv1d_39/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_39/conv1d/ExpandDims?
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_39_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_39/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_39/conv1d/ExpandDims_1/dim?
conv1d_39/conv1d/ExpandDims_1
ExpandDims4conv1d_39/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_39/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_39/conv1d/ExpandDims_1?
conv1d_39/conv1d	MLCConv2D$conv1d_39/conv1d/ExpandDims:output:0&conv1d_39/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_39/conv1d?
conv1d_39/conv1d/SqueezeSqueezeconv1d_39/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_39/conv1d/Squeeze?
 conv1d_39/BiasAdd/ReadVariableOpReadVariableOp)conv1d_39_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_39/BiasAdd/ReadVariableOp?
conv1d_39/BiasAddBiasAdd!conv1d_39/conv1d/Squeeze:output:0(conv1d_39/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_39/BiasAddz
conv1d_39/ReluReluconv1d_39/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_39/Relu?
conv1d_40/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_40/Pad/paddings?
conv1d_40/PadPadconv1d_39/Relu:activations:0conv1d_40/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_40/Pad?
conv1d_40/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_40/conv1d/ExpandDims/dim?
conv1d_40/conv1d/ExpandDims
ExpandDimsconv1d_40/Pad:output:0(conv1d_40/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_40/conv1d/ExpandDims?
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_40/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_40/conv1d/ExpandDims_1/dim?
conv1d_40/conv1d/ExpandDims_1
ExpandDims4conv1d_40/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_40/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_40/conv1d/ExpandDims_1?
conv1d_40/conv1d	MLCConv2D$conv1d_40/conv1d/ExpandDims:output:0&conv1d_40/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_40/conv1d?
conv1d_40/conv1d/SqueezeSqueezeconv1d_40/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_40/conv1d/Squeeze?
 conv1d_40/BiasAdd/ReadVariableOpReadVariableOp)conv1d_40_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_40/BiasAdd/ReadVariableOp?
conv1d_40/BiasAddBiasAdd!conv1d_40/conv1d/Squeeze:output:0(conv1d_40/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_40/BiasAddz
conv1d_40/ReluReluconv1d_40/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_40/Relu?
conv1d_41/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_41/Pad/paddings?
conv1d_41/PadPadconv1d_40/Relu:activations:0conv1d_41/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_41/Pad?
conv1d_41/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_41/conv1d/ExpandDims/dim?
conv1d_41/conv1d/ExpandDims
ExpandDimsconv1d_41/Pad:output:0(conv1d_41/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_41/conv1d/ExpandDims?
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_41/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_41/conv1d/ExpandDims_1/dim?
conv1d_41/conv1d/ExpandDims_1
ExpandDims4conv1d_41/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_41/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_41/conv1d/ExpandDims_1?
conv1d_41/conv1d	MLCConv2D$conv1d_41/conv1d/ExpandDims:output:0&conv1d_41/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_41/conv1d?
conv1d_41/conv1d/SqueezeSqueezeconv1d_41/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_41/conv1d/Squeeze?
 conv1d_41/BiasAdd/ReadVariableOpReadVariableOp)conv1d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_41/BiasAdd/ReadVariableOp?
conv1d_41/BiasAddBiasAdd!conv1d_41/conv1d/Squeeze:output:0(conv1d_41/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_41/BiasAddz
conv1d_41/ReluReluconv1d_41/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_41/Relus
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_5/Const?
flatten_5/ReshapeReshapeconv1d_41/Relu:activations:0flatten_5/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_5/Reshape?
!dense_10/MLCMatMul/ReadVariableOpReadVariableOp*dense_10_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_10/MLCMatMul/ReadVariableOp?
dense_10/MLCMatMul	MLCMatMulflatten_5/Reshape:output:0)dense_10/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/MLCMatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MLCMatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_10/Relu?
!dense_11/MLCMatMul/ReadVariableOpReadVariableOp*dense_11_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_11/MLCMatMul/ReadVariableOp?
dense_11/MLCMatMul	MLCMatMuldense_10/Relu:activations:0)dense_11/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/MLCMatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MLCMatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_11/Softmax?
IdentityIdentitydense_11/Softmax:softmax:0!^conv1d_35/BiasAdd/ReadVariableOp-^conv1d_35/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_36/BiasAdd/ReadVariableOp-^conv1d_36/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_37/BiasAdd/ReadVariableOp-^conv1d_37/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_38/BiasAdd/ReadVariableOp-^conv1d_38/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_39/BiasAdd/ReadVariableOp-^conv1d_39/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_40/BiasAdd/ReadVariableOp-^conv1d_40/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_41/BiasAdd/ReadVariableOp-^conv1d_41/conv1d/ExpandDims_1/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/MLCMatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2D
 conv1d_35/BiasAdd/ReadVariableOp conv1d_35/BiasAdd/ReadVariableOp2\
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOp,conv1d_35/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_36/BiasAdd/ReadVariableOp conv1d_36/BiasAdd/ReadVariableOp2\
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_37/BiasAdd/ReadVariableOp conv1d_37/BiasAdd/ReadVariableOp2\
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_38/BiasAdd/ReadVariableOp conv1d_38/BiasAdd/ReadVariableOp2\
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_39/BiasAdd/ReadVariableOp conv1d_39/BiasAdd/ReadVariableOp2\
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_40/BiasAdd/ReadVariableOp conv1d_40/BiasAdd/ReadVariableOp2\
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOp,conv1d_40/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_41/BiasAdd/ReadVariableOp conv1d_41/BiasAdd/ReadVariableOp2\
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOp,conv1d_41/conv1d/ExpandDims_1/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/MLCMatMul/ReadVariableOp!dense_10/MLCMatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/MLCMatMul/ReadVariableOp!dense_11/MLCMatMul/ReadVariableOp:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

*__inference_conv1d_35_layer_call_fn_701167

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
E__inference_conv1d_35_layer_call_and_return_conditional_losses_7002392
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
E__inference_conv1d_39_layer_call_and_return_conditional_losses_700375

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
*__inference_conv1d_40_layer_call_fn_701302

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
E__inference_conv1d_40_layer_call_and_return_conditional_losses_7004092
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
??
? 
"__inference__traced_restore_701792
file_prefix%
!assignvariableop_conv1d_35_kernel%
!assignvariableop_1_conv1d_35_bias'
#assignvariableop_2_conv1d_36_kernel%
!assignvariableop_3_conv1d_36_bias'
#assignvariableop_4_conv1d_37_kernel%
!assignvariableop_5_conv1d_37_bias'
#assignvariableop_6_conv1d_38_kernel%
!assignvariableop_7_conv1d_38_bias'
#assignvariableop_8_conv1d_39_kernel%
!assignvariableop_9_conv1d_39_bias(
$assignvariableop_10_conv1d_40_kernel&
"assignvariableop_11_conv1d_40_bias(
$assignvariableop_12_conv1d_41_kernel&
"assignvariableop_13_conv1d_41_bias'
#assignvariableop_14_dense_10_kernel%
!assignvariableop_15_dense_10_bias'
#assignvariableop_16_dense_11_kernel%
!assignvariableop_17_dense_11_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate
assignvariableop_23_total
assignvariableop_24_count
assignvariableop_25_total_1
assignvariableop_26_count_1/
+assignvariableop_27_adam_conv1d_35_kernel_m-
)assignvariableop_28_adam_conv1d_35_bias_m/
+assignvariableop_29_adam_conv1d_36_kernel_m-
)assignvariableop_30_adam_conv1d_36_bias_m/
+assignvariableop_31_adam_conv1d_37_kernel_m-
)assignvariableop_32_adam_conv1d_37_bias_m/
+assignvariableop_33_adam_conv1d_38_kernel_m-
)assignvariableop_34_adam_conv1d_38_bias_m/
+assignvariableop_35_adam_conv1d_39_kernel_m-
)assignvariableop_36_adam_conv1d_39_bias_m/
+assignvariableop_37_adam_conv1d_40_kernel_m-
)assignvariableop_38_adam_conv1d_40_bias_m/
+assignvariableop_39_adam_conv1d_41_kernel_m-
)assignvariableop_40_adam_conv1d_41_bias_m.
*assignvariableop_41_adam_dense_10_kernel_m,
(assignvariableop_42_adam_dense_10_bias_m.
*assignvariableop_43_adam_dense_11_kernel_m,
(assignvariableop_44_adam_dense_11_bias_m/
+assignvariableop_45_adam_conv1d_35_kernel_v-
)assignvariableop_46_adam_conv1d_35_bias_v/
+assignvariableop_47_adam_conv1d_36_kernel_v-
)assignvariableop_48_adam_conv1d_36_bias_v/
+assignvariableop_49_adam_conv1d_37_kernel_v-
)assignvariableop_50_adam_conv1d_37_bias_v/
+assignvariableop_51_adam_conv1d_38_kernel_v-
)assignvariableop_52_adam_conv1d_38_bias_v/
+assignvariableop_53_adam_conv1d_39_kernel_v-
)assignvariableop_54_adam_conv1d_39_bias_v/
+assignvariableop_55_adam_conv1d_40_kernel_v-
)assignvariableop_56_adam_conv1d_40_bias_v/
+assignvariableop_57_adam_conv1d_41_kernel_v-
)assignvariableop_58_adam_conv1d_41_bias_v.
*assignvariableop_59_adam_dense_10_kernel_v,
(assignvariableop_60_adam_dense_10_bias_v.
*assignvariableop_61_adam_dense_11_kernel_v,
(assignvariableop_62_adam_dense_11_bias_v
identity_64??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?"
value?"B?"@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?
value?B?@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_35_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_35_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv1d_36_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv1d_36_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv1d_37_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv1d_37_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv1d_38_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv1d_38_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv1d_39_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv1d_39_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv1d_40_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv1d_40_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv1d_41_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv1d_41_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_10_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_10_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_11_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_11_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv1d_35_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv1d_35_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv1d_36_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv1d_36_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv1d_37_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv1d_37_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv1d_38_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv1d_38_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv1d_39_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv1d_39_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv1d_40_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv1d_40_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv1d_41_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv1d_41_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_10_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_10_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_11_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_11_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv1d_35_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv1d_35_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv1d_36_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv1d_36_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv1d_37_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv1d_37_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv1d_38_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv1d_38_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv1d_39_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv1d_39_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv1d_40_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv1d_40_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv1d_41_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv1d_41_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_dense_10_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_dense_10_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_11_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_11_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_629
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_63?
Identity_64IdentityIdentity_63:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_64"#
identity_64Identity_64:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
F
*__inference_flatten_5_layer_call_fn_701340

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
E__inference_flatten_5_layer_call_and_return_conditional_losses_7004652
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
?
?
(__inference_model_5_layer_call_fn_700766
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

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconvcatunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_7007272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
?

?
D__inference_dense_10_layer_call_and_return_conditional_losses_701351

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?
?
E__inference_conv1d_40_layer_call_and_return_conditional_losses_701293

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
*__inference_conv1d_36_layer_call_fn_701194

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
E__inference_conv1d_36_layer_call_and_return_conditional_losses_7002732
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
E__inference_conv1d_41_layer_call_and_return_conditional_losses_700443

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
(__inference_model_5_layer_call_fn_701140
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

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_7007272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

*__inference_conv1d_38_layer_call_fn_701248

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
E__inference_conv1d_38_layer_call_and_return_conditional_losses_7003412
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
E__inference_conv1d_41_layer_call_and_return_conditional_losses_701320

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
E__inference_conv1d_37_layer_call_and_return_conditional_losses_701212

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
?
$__inference_signature_wrapper_700818
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

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconvcatunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_7002162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:?????????

_user_specified_namecat:QM
+
_output_shapes
:?????????

_user_specified_nameconv
?
?
E__inference_conv1d_40_layer_call_and_return_conditional_losses_700409

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
E__inference_conv1d_35_layer_call_and_return_conditional_losses_701158

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
?
?
E__inference_conv1d_38_layer_call_and_return_conditional_losses_701239

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
?3
?
C__inference_model_5_layer_call_and_return_conditional_losses_700579
conv
cat
conv1d_35_700532
conv1d_35_700534
conv1d_36_700537
conv1d_36_700539
conv1d_37_700542
conv1d_37_700544
conv1d_38_700547
conv1d_38_700549
conv1d_39_700552
conv1d_39_700554
conv1d_40_700557
conv1d_40_700559
conv1d_41_700562
conv1d_41_700564
dense_10_700568
dense_10_700570
dense_11_700573
dense_11_700575
identity??!conv1d_35/StatefulPartitionedCall?!conv1d_36/StatefulPartitionedCall?!conv1d_37/StatefulPartitionedCall?!conv1d_38/StatefulPartitionedCall?!conv1d_39/StatefulPartitionedCall?!conv1d_40/StatefulPartitionedCall?!conv1d_41/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
!conv1d_35/StatefulPartitionedCallStatefulPartitionedCallconvconv1d_35_700532conv1d_35_700534*
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
E__inference_conv1d_35_layer_call_and_return_conditional_losses_7002392#
!conv1d_35/StatefulPartitionedCall?
!conv1d_36/StatefulPartitionedCallStatefulPartitionedCall*conv1d_35/StatefulPartitionedCall:output:0conv1d_36_700537conv1d_36_700539*
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
E__inference_conv1d_36_layer_call_and_return_conditional_losses_7002732#
!conv1d_36/StatefulPartitionedCall?
!conv1d_37/StatefulPartitionedCallStatefulPartitionedCall*conv1d_36/StatefulPartitionedCall:output:0conv1d_37_700542conv1d_37_700544*
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
E__inference_conv1d_37_layer_call_and_return_conditional_losses_7003072#
!conv1d_37/StatefulPartitionedCall?
!conv1d_38/StatefulPartitionedCallStatefulPartitionedCall*conv1d_37/StatefulPartitionedCall:output:0conv1d_38_700547conv1d_38_700549*
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
E__inference_conv1d_38_layer_call_and_return_conditional_losses_7003412#
!conv1d_38/StatefulPartitionedCall?
!conv1d_39/StatefulPartitionedCallStatefulPartitionedCall*conv1d_38/StatefulPartitionedCall:output:0conv1d_39_700552conv1d_39_700554*
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
E__inference_conv1d_39_layer_call_and_return_conditional_losses_7003752#
!conv1d_39/StatefulPartitionedCall?
!conv1d_40/StatefulPartitionedCallStatefulPartitionedCall*conv1d_39/StatefulPartitionedCall:output:0conv1d_40_700557conv1d_40_700559*
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
E__inference_conv1d_40_layer_call_and_return_conditional_losses_7004092#
!conv1d_40/StatefulPartitionedCall?
!conv1d_41/StatefulPartitionedCallStatefulPartitionedCall*conv1d_40/StatefulPartitionedCall:output:0conv1d_41_700562conv1d_41_700564*
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
E__inference_conv1d_41_layer_call_and_return_conditional_losses_7004432#
!conv1d_41/StatefulPartitionedCall?
flatten_5/PartitionedCallPartitionedCall*conv1d_41/StatefulPartitionedCall:output:0*
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_7004652
flatten_5/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_10_700568dense_10_700570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_7004842"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_700573dense_11_700575*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_7005112"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0"^conv1d_35/StatefulPartitionedCall"^conv1d_36/StatefulPartitionedCall"^conv1d_37/StatefulPartitionedCall"^conv1d_38/StatefulPartitionedCall"^conv1d_39/StatefulPartitionedCall"^conv1d_40/StatefulPartitionedCall"^conv1d_41/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2F
!conv1d_35/StatefulPartitionedCall!conv1d_35/StatefulPartitionedCall2F
!conv1d_36/StatefulPartitionedCall!conv1d_36/StatefulPartitionedCall2F
!conv1d_37/StatefulPartitionedCall!conv1d_37/StatefulPartitionedCall2F
!conv1d_38/StatefulPartitionedCall!conv1d_38/StatefulPartitionedCall2F
!conv1d_39/StatefulPartitionedCall!conv1d_39/StatefulPartitionedCall2F
!conv1d_40/StatefulPartitionedCall!conv1d_40/StatefulPartitionedCall2F
!conv1d_41/StatefulPartitionedCall!conv1d_41/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
??
?
!__inference__wrapped_model_700216
conv
catA
=model_5_conv1d_35_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_35_biasadd_readvariableop_resourceA
=model_5_conv1d_36_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_36_biasadd_readvariableop_resourceA
=model_5_conv1d_37_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_37_biasadd_readvariableop_resourceA
=model_5_conv1d_38_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_38_biasadd_readvariableop_resourceA
=model_5_conv1d_39_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_39_biasadd_readvariableop_resourceA
=model_5_conv1d_40_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_40_biasadd_readvariableop_resourceA
=model_5_conv1d_41_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_41_biasadd_readvariableop_resource6
2model_5_dense_10_mlcmatmul_readvariableop_resource4
0model_5_dense_10_biasadd_readvariableop_resource6
2model_5_dense_11_mlcmatmul_readvariableop_resource4
0model_5_dense_11_biasadd_readvariableop_resource
identity??(model_5/conv1d_35/BiasAdd/ReadVariableOp?4model_5/conv1d_35/conv1d/ExpandDims_1/ReadVariableOp?(model_5/conv1d_36/BiasAdd/ReadVariableOp?4model_5/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp?(model_5/conv1d_37/BiasAdd/ReadVariableOp?4model_5/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp?(model_5/conv1d_38/BiasAdd/ReadVariableOp?4model_5/conv1d_38/conv1d/ExpandDims_1/ReadVariableOp?(model_5/conv1d_39/BiasAdd/ReadVariableOp?4model_5/conv1d_39/conv1d/ExpandDims_1/ReadVariableOp?(model_5/conv1d_40/BiasAdd/ReadVariableOp?4model_5/conv1d_40/conv1d/ExpandDims_1/ReadVariableOp?(model_5/conv1d_41/BiasAdd/ReadVariableOp?4model_5/conv1d_41/conv1d/ExpandDims_1/ReadVariableOp?'model_5/dense_10/BiasAdd/ReadVariableOp?)model_5/dense_10/MLCMatMul/ReadVariableOp?'model_5/dense_11/BiasAdd/ReadVariableOp?)model_5/dense_11/MLCMatMul/ReadVariableOp?
model_5/conv1d_35/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_5/conv1d_35/Pad/paddings?
model_5/conv1d_35/PadPadconv'model_5/conv1d_35/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
model_5/conv1d_35/Pad?
'model_5/conv1d_35/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_5/conv1d_35/conv1d/ExpandDims/dim?
#model_5/conv1d_35/conv1d/ExpandDims
ExpandDimsmodel_5/conv1d_35/Pad:output:00model_5/conv1d_35/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2%
#model_5/conv1d_35/conv1d/ExpandDims?
4model_5/conv1d_35/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_35_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype026
4model_5/conv1d_35/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_35/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_35/conv1d/ExpandDims_1/dim?
%model_5/conv1d_35/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_35/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_35/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2'
%model_5/conv1d_35/conv1d/ExpandDims_1?
model_5/conv1d_35/conv1d	MLCConv2D,model_5/conv1d_35/conv1d/ExpandDims:output:0.model_5/conv1d_35/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_5/conv1d_35/conv1d?
 model_5/conv1d_35/conv1d/SqueezeSqueeze!model_5/conv1d_35/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_5/conv1d_35/conv1d/Squeeze?
(model_5/conv1d_35/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_5/conv1d_35/BiasAdd/ReadVariableOp?
model_5/conv1d_35/BiasAddBiasAdd)model_5/conv1d_35/conv1d/Squeeze:output:00model_5/conv1d_35/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_35/BiasAdd?
model_5/conv1d_35/ReluRelu"model_5/conv1d_35/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_35/Relu?
model_5/conv1d_36/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_5/conv1d_36/Pad/paddings?
model_5/conv1d_36/PadPad$model_5/conv1d_35/Relu:activations:0'model_5/conv1d_36/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_36/Pad?
'model_5/conv1d_36/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_5/conv1d_36/conv1d/ExpandDims/dim?
#model_5/conv1d_36/conv1d/ExpandDims
ExpandDimsmodel_5/conv1d_36/Pad:output:00model_5/conv1d_36/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_5/conv1d_36/conv1d/ExpandDims?
4model_5/conv1d_36/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_36_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_5/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_36/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_36/conv1d/ExpandDims_1/dim?
%model_5/conv1d_36/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_36/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_5/conv1d_36/conv1d/ExpandDims_1?
model_5/conv1d_36/conv1d	MLCConv2D,model_5/conv1d_36/conv1d/ExpandDims:output:0.model_5/conv1d_36/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_5/conv1d_36/conv1d?
 model_5/conv1d_36/conv1d/SqueezeSqueeze!model_5/conv1d_36/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_5/conv1d_36/conv1d/Squeeze?
(model_5/conv1d_36/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_36_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_5/conv1d_36/BiasAdd/ReadVariableOp?
model_5/conv1d_36/BiasAddBiasAdd)model_5/conv1d_36/conv1d/Squeeze:output:00model_5/conv1d_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_36/BiasAdd?
model_5/conv1d_36/ReluRelu"model_5/conv1d_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_36/Relu?
model_5/conv1d_37/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_5/conv1d_37/Pad/paddings?
model_5/conv1d_37/PadPad$model_5/conv1d_36/Relu:activations:0'model_5/conv1d_37/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_37/Pad?
'model_5/conv1d_37/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_5/conv1d_37/conv1d/ExpandDims/dim?
#model_5/conv1d_37/conv1d/ExpandDims
ExpandDimsmodel_5/conv1d_37/Pad:output:00model_5/conv1d_37/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_5/conv1d_37/conv1d/ExpandDims?
4model_5/conv1d_37/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_37_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_5/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_37/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_37/conv1d/ExpandDims_1/dim?
%model_5/conv1d_37/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_37/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_5/conv1d_37/conv1d/ExpandDims_1?
model_5/conv1d_37/conv1d	MLCConv2D,model_5/conv1d_37/conv1d/ExpandDims:output:0.model_5/conv1d_37/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_5/conv1d_37/conv1d?
 model_5/conv1d_37/conv1d/SqueezeSqueeze!model_5/conv1d_37/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_5/conv1d_37/conv1d/Squeeze?
(model_5/conv1d_37/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_5/conv1d_37/BiasAdd/ReadVariableOp?
model_5/conv1d_37/BiasAddBiasAdd)model_5/conv1d_37/conv1d/Squeeze:output:00model_5/conv1d_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_37/BiasAdd?
model_5/conv1d_37/ReluRelu"model_5/conv1d_37/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_37/Relu?
model_5/conv1d_38/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_5/conv1d_38/Pad/paddings?
model_5/conv1d_38/PadPad$model_5/conv1d_37/Relu:activations:0'model_5/conv1d_38/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_38/Pad?
'model_5/conv1d_38/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_5/conv1d_38/conv1d/ExpandDims/dim?
#model_5/conv1d_38/conv1d/ExpandDims
ExpandDimsmodel_5/conv1d_38/Pad:output:00model_5/conv1d_38/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_5/conv1d_38/conv1d/ExpandDims?
4model_5/conv1d_38/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_38_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_5/conv1d_38/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_38/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_38/conv1d/ExpandDims_1/dim?
%model_5/conv1d_38/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_38/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_38/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_5/conv1d_38/conv1d/ExpandDims_1?
model_5/conv1d_38/conv1d	MLCConv2D,model_5/conv1d_38/conv1d/ExpandDims:output:0.model_5/conv1d_38/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_5/conv1d_38/conv1d?
 model_5/conv1d_38/conv1d/SqueezeSqueeze!model_5/conv1d_38/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_5/conv1d_38/conv1d/Squeeze?
(model_5/conv1d_38/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_38_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_5/conv1d_38/BiasAdd/ReadVariableOp?
model_5/conv1d_38/BiasAddBiasAdd)model_5/conv1d_38/conv1d/Squeeze:output:00model_5/conv1d_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_38/BiasAdd?
model_5/conv1d_38/ReluRelu"model_5/conv1d_38/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_38/Relu?
model_5/conv1d_39/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_5/conv1d_39/Pad/paddings?
model_5/conv1d_39/PadPad$model_5/conv1d_38/Relu:activations:0'model_5/conv1d_39/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_39/Pad?
'model_5/conv1d_39/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_5/conv1d_39/conv1d/ExpandDims/dim?
#model_5/conv1d_39/conv1d/ExpandDims
ExpandDimsmodel_5/conv1d_39/Pad:output:00model_5/conv1d_39/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_5/conv1d_39/conv1d/ExpandDims?
4model_5/conv1d_39/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_39_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_5/conv1d_39/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_39/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_39/conv1d/ExpandDims_1/dim?
%model_5/conv1d_39/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_39/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_39/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_5/conv1d_39/conv1d/ExpandDims_1?
model_5/conv1d_39/conv1d	MLCConv2D,model_5/conv1d_39/conv1d/ExpandDims:output:0.model_5/conv1d_39/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_5/conv1d_39/conv1d?
 model_5/conv1d_39/conv1d/SqueezeSqueeze!model_5/conv1d_39/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_5/conv1d_39/conv1d/Squeeze?
(model_5/conv1d_39/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_39_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_5/conv1d_39/BiasAdd/ReadVariableOp?
model_5/conv1d_39/BiasAddBiasAdd)model_5/conv1d_39/conv1d/Squeeze:output:00model_5/conv1d_39/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_39/BiasAdd?
model_5/conv1d_39/ReluRelu"model_5/conv1d_39/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_39/Relu?
model_5/conv1d_40/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_5/conv1d_40/Pad/paddings?
model_5/conv1d_40/PadPad$model_5/conv1d_39/Relu:activations:0'model_5/conv1d_40/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_40/Pad?
'model_5/conv1d_40/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_5/conv1d_40/conv1d/ExpandDims/dim?
#model_5/conv1d_40/conv1d/ExpandDims
ExpandDimsmodel_5/conv1d_40/Pad:output:00model_5/conv1d_40/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_5/conv1d_40/conv1d/ExpandDims?
4model_5/conv1d_40/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_40_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_5/conv1d_40/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_40/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_40/conv1d/ExpandDims_1/dim?
%model_5/conv1d_40/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_40/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_40/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_5/conv1d_40/conv1d/ExpandDims_1?
model_5/conv1d_40/conv1d	MLCConv2D,model_5/conv1d_40/conv1d/ExpandDims:output:0.model_5/conv1d_40/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_5/conv1d_40/conv1d?
 model_5/conv1d_40/conv1d/SqueezeSqueeze!model_5/conv1d_40/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_5/conv1d_40/conv1d/Squeeze?
(model_5/conv1d_40/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_40_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_5/conv1d_40/BiasAdd/ReadVariableOp?
model_5/conv1d_40/BiasAddBiasAdd)model_5/conv1d_40/conv1d/Squeeze:output:00model_5/conv1d_40/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_40/BiasAdd?
model_5/conv1d_40/ReluRelu"model_5/conv1d_40/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_40/Relu?
model_5/conv1d_41/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_5/conv1d_41/Pad/paddings?
model_5/conv1d_41/PadPad$model_5/conv1d_40/Relu:activations:0'model_5/conv1d_41/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_41/Pad?
'model_5/conv1d_41/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_5/conv1d_41/conv1d/ExpandDims/dim?
#model_5/conv1d_41/conv1d/ExpandDims
ExpandDimsmodel_5/conv1d_41/Pad:output:00model_5/conv1d_41/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_5/conv1d_41/conv1d/ExpandDims?
4model_5/conv1d_41/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_41_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_5/conv1d_41/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_41/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_41/conv1d/ExpandDims_1/dim?
%model_5/conv1d_41/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_41/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_41/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_5/conv1d_41/conv1d/ExpandDims_1?
model_5/conv1d_41/conv1d	MLCConv2D,model_5/conv1d_41/conv1d/ExpandDims:output:0.model_5/conv1d_41/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_5/conv1d_41/conv1d?
 model_5/conv1d_41/conv1d/SqueezeSqueeze!model_5/conv1d_41/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_5/conv1d_41/conv1d/Squeeze?
(model_5/conv1d_41/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_5/conv1d_41/BiasAdd/ReadVariableOp?
model_5/conv1d_41/BiasAddBiasAdd)model_5/conv1d_41/conv1d/Squeeze:output:00model_5/conv1d_41/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_41/BiasAdd?
model_5/conv1d_41/ReluRelu"model_5/conv1d_41/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_5/conv1d_41/Relu?
model_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model_5/flatten_5/Const?
model_5/flatten_5/ReshapeReshape$model_5/conv1d_41/Relu:activations:0 model_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:??????????2
model_5/flatten_5/Reshape?
)model_5/dense_10/MLCMatMul/ReadVariableOpReadVariableOp2model_5_dense_10_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)model_5/dense_10/MLCMatMul/ReadVariableOp?
model_5/dense_10/MLCMatMul	MLCMatMul"model_5/flatten_5/Reshape:output:01model_5/dense_10/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_5/dense_10/MLCMatMul?
'model_5/dense_10/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_5/dense_10/BiasAdd/ReadVariableOp?
model_5/dense_10/BiasAddBiasAdd$model_5/dense_10/MLCMatMul:product:0/model_5/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_5/dense_10/BiasAdd?
model_5/dense_10/ReluRelu!model_5/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_5/dense_10/Relu?
)model_5/dense_11/MLCMatMul/ReadVariableOpReadVariableOp2model_5_dense_11_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)model_5/dense_11/MLCMatMul/ReadVariableOp?
model_5/dense_11/MLCMatMul	MLCMatMul#model_5/dense_10/Relu:activations:01model_5/dense_11/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_5/dense_11/MLCMatMul?
'model_5/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_5/dense_11/BiasAdd/ReadVariableOp?
model_5/dense_11/BiasAddBiasAdd$model_5/dense_11/MLCMatMul:product:0/model_5/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_5/dense_11/BiasAdd?
model_5/dense_11/SoftmaxSoftmax!model_5/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_5/dense_11/Softmax?
IdentityIdentity"model_5/dense_11/Softmax:softmax:0)^model_5/conv1d_35/BiasAdd/ReadVariableOp5^model_5/conv1d_35/conv1d/ExpandDims_1/ReadVariableOp)^model_5/conv1d_36/BiasAdd/ReadVariableOp5^model_5/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp)^model_5/conv1d_37/BiasAdd/ReadVariableOp5^model_5/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp)^model_5/conv1d_38/BiasAdd/ReadVariableOp5^model_5/conv1d_38/conv1d/ExpandDims_1/ReadVariableOp)^model_5/conv1d_39/BiasAdd/ReadVariableOp5^model_5/conv1d_39/conv1d/ExpandDims_1/ReadVariableOp)^model_5/conv1d_40/BiasAdd/ReadVariableOp5^model_5/conv1d_40/conv1d/ExpandDims_1/ReadVariableOp)^model_5/conv1d_41/BiasAdd/ReadVariableOp5^model_5/conv1d_41/conv1d/ExpandDims_1/ReadVariableOp(^model_5/dense_10/BiasAdd/ReadVariableOp*^model_5/dense_10/MLCMatMul/ReadVariableOp(^model_5/dense_11/BiasAdd/ReadVariableOp*^model_5/dense_11/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2T
(model_5/conv1d_35/BiasAdd/ReadVariableOp(model_5/conv1d_35/BiasAdd/ReadVariableOp2l
4model_5/conv1d_35/conv1d/ExpandDims_1/ReadVariableOp4model_5/conv1d_35/conv1d/ExpandDims_1/ReadVariableOp2T
(model_5/conv1d_36/BiasAdd/ReadVariableOp(model_5/conv1d_36/BiasAdd/ReadVariableOp2l
4model_5/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp4model_5/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp2T
(model_5/conv1d_37/BiasAdd/ReadVariableOp(model_5/conv1d_37/BiasAdd/ReadVariableOp2l
4model_5/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp4model_5/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp2T
(model_5/conv1d_38/BiasAdd/ReadVariableOp(model_5/conv1d_38/BiasAdd/ReadVariableOp2l
4model_5/conv1d_38/conv1d/ExpandDims_1/ReadVariableOp4model_5/conv1d_38/conv1d/ExpandDims_1/ReadVariableOp2T
(model_5/conv1d_39/BiasAdd/ReadVariableOp(model_5/conv1d_39/BiasAdd/ReadVariableOp2l
4model_5/conv1d_39/conv1d/ExpandDims_1/ReadVariableOp4model_5/conv1d_39/conv1d/ExpandDims_1/ReadVariableOp2T
(model_5/conv1d_40/BiasAdd/ReadVariableOp(model_5/conv1d_40/BiasAdd/ReadVariableOp2l
4model_5/conv1d_40/conv1d/ExpandDims_1/ReadVariableOp4model_5/conv1d_40/conv1d/ExpandDims_1/ReadVariableOp2T
(model_5/conv1d_41/BiasAdd/ReadVariableOp(model_5/conv1d_41/BiasAdd/ReadVariableOp2l
4model_5/conv1d_41/conv1d/ExpandDims_1/ReadVariableOp4model_5/conv1d_41/conv1d/ExpandDims_1/ReadVariableOp2R
'model_5/dense_10/BiasAdd/ReadVariableOp'model_5/dense_10/BiasAdd/ReadVariableOp2V
)model_5/dense_10/MLCMatMul/ReadVariableOp)model_5/dense_10/MLCMatMul/ReadVariableOp2R
'model_5/dense_11/BiasAdd/ReadVariableOp'model_5/dense_11/BiasAdd/ReadVariableOp2V
)model_5/dense_11/MLCMatMul/ReadVariableOp)model_5/dense_11/MLCMatMul/ReadVariableOp:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
?

?
D__inference_dense_10_layer_call_and_return_conditional_losses_700484

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?
?
E__inference_conv1d_36_layer_call_and_return_conditional_losses_701185

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
?
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_700465

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
?
?
E__inference_conv1d_37_layer_call_and_return_conditional_losses_700307

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
*__inference_conv1d_39_layer_call_fn_701275

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
E__inference_conv1d_39_layer_call_and_return_conditional_losses_7003752
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
serving_default_cat:0?????????
9
conv1
serving_default_conv:0?????????<
dense_110
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?{
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
layer_with_weights-6
layer-7
	layer-8

layer_with_weights-7

layer-9
layer-10
layer_with_weights-8
layer-11
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?v
_tf_keras_network?v{"class_name": "Functional", "name": "model_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv"}, "name": "conv", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_35", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_35", "inbound_nodes": [[["conv", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_36", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_36", "inbound_nodes": [[["conv1d_35", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_37", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_37", "inbound_nodes": [[["conv1d_36", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_38", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_38", "inbound_nodes": [[["conv1d_37", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_39", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_39", "inbound_nodes": [[["conv1d_38", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_40", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_40", "inbound_nodes": [[["conv1d_39", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_41", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_41", "inbound_nodes": [[["conv1d_40", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["conv1d_41", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["flatten_5", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cat"}, "name": "cat", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}], "input_layers": [["conv", 0, 0], ["cat", 0, 0]], "output_layers": [["dense_11", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 26, 1]}, {"class_name": "TensorShape", "items": [null, 3]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv"}, "name": "conv", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_35", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_35", "inbound_nodes": [[["conv", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_36", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_36", "inbound_nodes": [[["conv1d_35", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_37", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_37", "inbound_nodes": [[["conv1d_36", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_38", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_38", "inbound_nodes": [[["conv1d_37", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_39", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_39", "inbound_nodes": [[["conv1d_38", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_40", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_40", "inbound_nodes": [[["conv1d_39", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_41", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_41", "inbound_nodes": [[["conv1d_40", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["conv1d_41", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["flatten_5", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cat"}, "name": "cat", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}], "input_layers": [["conv", 0, 0], ["cat", 0, 0]], "output_layers": [["dense_11", 0, 0]]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv"}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv1D", "name": "conv1d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_35", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 1]}}
?	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_36", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_37", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_38", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_39", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_40", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

7kernel
8bias
9trainable_variables
:regularization_losses
;	variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_41", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?
=trainable_variables
>regularization_losses
?	variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Akernel
Bbias
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1664}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1664]}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cat", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cat"}}
?

Gkernel
Hbias
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17]}}
?
Miter

Nbeta_1

Obeta_2
	Pdecay
Qlearning_ratem?m?m?m?m? m?%m?&m?+m?,m?1m?2m?7m?8m?Am?Bm?Gm?Hm?v?v?v?v?v? v?%v?&v?+v?,v?1v?2v?7v?8v?Av?Bv?Gv?Hv?"
	optimizer
?
0
1
2
3
4
 5
%6
&7
+8
,9
110
211
712
813
A14
B15
G16
H17"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
 5
%6
&7
+8
,9
110
211
712
813
A14
B15
G16
H17"
trackable_list_wrapper
?
trainable_variables
regularization_losses
Rlayer_regularization_losses
Snon_trainable_variables
Tlayer_metrics
	variables
Umetrics

Vlayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
&:$@2conv1d_35/kernel
:@2conv1d_35/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
regularization_losses
Wlayer_regularization_losses
Xlayer_metrics
Ynon_trainable_variables
	variables
Zmetrics

[layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_36/kernel
:@2conv1d_36/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
regularization_losses
\layer_regularization_losses
]layer_metrics
^non_trainable_variables
	variables
_metrics

`layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_37/kernel
:@2conv1d_37/bias
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
!trainable_variables
"regularization_losses
alayer_regularization_losses
blayer_metrics
cnon_trainable_variables
#	variables
dmetrics

elayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_38/kernel
:@2conv1d_38/bias
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
'trainable_variables
(regularization_losses
flayer_regularization_losses
glayer_metrics
hnon_trainable_variables
)	variables
imetrics

jlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_39/kernel
:@2conv1d_39/bias
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
-trainable_variables
.regularization_losses
klayer_regularization_losses
llayer_metrics
mnon_trainable_variables
/	variables
nmetrics

olayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_40/kernel
:@2conv1d_40/bias
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
3trainable_variables
4regularization_losses
player_regularization_losses
qlayer_metrics
rnon_trainable_variables
5	variables
smetrics

tlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_41/kernel
:@2conv1d_41/bias
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
9trainable_variables
:regularization_losses
ulayer_regularization_losses
vlayer_metrics
wnon_trainable_variables
;	variables
xmetrics

ylayers
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
=trainable_variables
>regularization_losses
zlayer_regularization_losses
{layer_metrics
|non_trainable_variables
?	variables
}metrics

~layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_10/kernel
:2dense_10/bias
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
Ctrainable_variables
Dregularization_losses
layer_regularization_losses
?layer_metrics
?non_trainable_variables
E	variables
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_11/kernel
:2dense_11/bias
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
?
Itrainable_variables
Jregularization_losses
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
K	variables
?metrics
?layers
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
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
v
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
11"
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
+:)@2Adam/conv1d_35/kernel/m
!:@2Adam/conv1d_35/bias/m
+:)@@2Adam/conv1d_36/kernel/m
!:@2Adam/conv1d_36/bias/m
+:)@@2Adam/conv1d_37/kernel/m
!:@2Adam/conv1d_37/bias/m
+:)@@2Adam/conv1d_38/kernel/m
!:@2Adam/conv1d_38/bias/m
+:)@@2Adam/conv1d_39/kernel/m
!:@2Adam/conv1d_39/bias/m
+:)@@2Adam/conv1d_40/kernel/m
!:@2Adam/conv1d_40/bias/m
+:)@@2Adam/conv1d_41/kernel/m
!:@2Adam/conv1d_41/bias/m
':%	?2Adam/dense_10/kernel/m
 :2Adam/dense_10/bias/m
&:$2Adam/dense_11/kernel/m
 :2Adam/dense_11/bias/m
+:)@2Adam/conv1d_35/kernel/v
!:@2Adam/conv1d_35/bias/v
+:)@@2Adam/conv1d_36/kernel/v
!:@2Adam/conv1d_36/bias/v
+:)@@2Adam/conv1d_37/kernel/v
!:@2Adam/conv1d_37/bias/v
+:)@@2Adam/conv1d_38/kernel/v
!:@2Adam/conv1d_38/bias/v
+:)@@2Adam/conv1d_39/kernel/v
!:@2Adam/conv1d_39/bias/v
+:)@@2Adam/conv1d_40/kernel/v
!:@2Adam/conv1d_40/bias/v
+:)@@2Adam/conv1d_41/kernel/v
!:@2Adam/conv1d_41/bias/v
':%	?2Adam/dense_10/kernel/v
 :2Adam/dense_10/bias/v
&:$2Adam/dense_11/kernel/v
 :2Adam/dense_11/bias/v
?2?
(__inference_model_5_layer_call_fn_700766
(__inference_model_5_layer_call_fn_701140
(__inference_model_5_layer_call_fn_701098
(__inference_model_5_layer_call_fn_700673?
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
!__inference__wrapped_model_700216?
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
cat?????????
?2?
C__inference_model_5_layer_call_and_return_conditional_losses_700937
C__inference_model_5_layer_call_and_return_conditional_losses_700579
C__inference_model_5_layer_call_and_return_conditional_losses_701056
C__inference_model_5_layer_call_and_return_conditional_losses_700528?
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
*__inference_conv1d_35_layer_call_fn_701167?
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
E__inference_conv1d_35_layer_call_and_return_conditional_losses_701158?
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
*__inference_conv1d_36_layer_call_fn_701194?
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
E__inference_conv1d_36_layer_call_and_return_conditional_losses_701185?
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
*__inference_conv1d_37_layer_call_fn_701221?
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
E__inference_conv1d_37_layer_call_and_return_conditional_losses_701212?
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
*__inference_conv1d_38_layer_call_fn_701248?
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
E__inference_conv1d_38_layer_call_and_return_conditional_losses_701239?
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
*__inference_conv1d_39_layer_call_fn_701275?
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
E__inference_conv1d_39_layer_call_and_return_conditional_losses_701266?
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
*__inference_conv1d_40_layer_call_fn_701302?
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
E__inference_conv1d_40_layer_call_and_return_conditional_losses_701293?
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
*__inference_conv1d_41_layer_call_fn_701329?
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
E__inference_conv1d_41_layer_call_and_return_conditional_losses_701320?
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
*__inference_flatten_5_layer_call_fn_701340?
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_701335?
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
)__inference_dense_10_layer_call_fn_701360?
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
D__inference_dense_10_layer_call_and_return_conditional_losses_701351?
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
)__inference_dense_11_layer_call_fn_701380?
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
D__inference_dense_11_layer_call_and_return_conditional_losses_701371?
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
$__inference_signature_wrapper_700818catconv"?
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
!__inference__wrapped_model_700216? %&+,1278ABGHU?R
K?H
F?C
"?
conv?????????
?
cat?????????
? "3?0
.
dense_11"?
dense_11??????????
E__inference_conv1d_35_layer_call_and_return_conditional_losses_701158d3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????@
? ?
*__inference_conv1d_35_layer_call_fn_701167W3?0
)?&
$?!
inputs?????????
? "??????????@?
E__inference_conv1d_36_layer_call_and_return_conditional_losses_701185d3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_36_layer_call_fn_701194W3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_37_layer_call_and_return_conditional_losses_701212d 3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_37_layer_call_fn_701221W 3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_38_layer_call_and_return_conditional_losses_701239d%&3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_38_layer_call_fn_701248W%&3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_39_layer_call_and_return_conditional_losses_701266d+,3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_39_layer_call_fn_701275W+,3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_40_layer_call_and_return_conditional_losses_701293d123?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_40_layer_call_fn_701302W123?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_41_layer_call_and_return_conditional_losses_701320d783?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_41_layer_call_fn_701329W783?0
)?&
$?!
inputs?????????@
? "??????????@?
D__inference_dense_10_layer_call_and_return_conditional_losses_701351]AB0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_10_layer_call_fn_701360PAB0?-
&?#
!?
inputs??????????
? "???????????
D__inference_dense_11_layer_call_and_return_conditional_losses_701371\GH/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_11_layer_call_fn_701380OGH/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_5_layer_call_and_return_conditional_losses_701335]3?0
)?&
$?!
inputs?????????@
? "&?#
?
0??????????
? ~
*__inference_flatten_5_layer_call_fn_701340P3?0
)?&
$?!
inputs?????????@
? "????????????
C__inference_model_5_layer_call_and_return_conditional_losses_700528? %&+,1278ABGH]?Z
S?P
F?C
"?
conv?????????
?
cat?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_model_5_layer_call_and_return_conditional_losses_700579? %&+,1278ABGH]?Z
S?P
F?C
"?
conv?????????
?
cat?????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_5_layer_call_and_return_conditional_losses_700937? %&+,1278ABGHf?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_model_5_layer_call_and_return_conditional_losses_701056? %&+,1278ABGHf?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
(__inference_model_5_layer_call_fn_700673? %&+,1278ABGH]?Z
S?P
F?C
"?
conv?????????
?
cat?????????
p

 
? "???????????
(__inference_model_5_layer_call_fn_700766? %&+,1278ABGH]?Z
S?P
F?C
"?
conv?????????
?
cat?????????
p 

 
? "???????????
(__inference_model_5_layer_call_fn_701098? %&+,1278ABGHf?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
(__inference_model_5_layer_call_fn_701140? %&+,1278ABGHf?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
$__inference_signature_wrapper_700818? %&+,1278ABGH_?\
? 
U?R
$
cat?
cat?????????
*
conv"?
conv?????????"3?0
.
dense_11"?
dense_11?????????