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
conv1d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv1d_28/kernel
y
$conv1d_28/kernel/Read/ReadVariableOpReadVariableOpconv1d_28/kernel*"
_output_shapes
:@*
dtype0
t
conv1d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_28/bias
m
"conv1d_28/bias/Read/ReadVariableOpReadVariableOpconv1d_28/bias*
_output_shapes
:@*
dtype0
?
conv1d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_29/kernel
y
$conv1d_29/kernel/Read/ReadVariableOpReadVariableOpconv1d_29/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_29/bias
m
"conv1d_29/bias/Read/ReadVariableOpReadVariableOpconv1d_29/bias*
_output_shapes
:@*
dtype0
?
conv1d_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_30/kernel
y
$conv1d_30/kernel/Read/ReadVariableOpReadVariableOpconv1d_30/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_30/bias
m
"conv1d_30/bias/Read/ReadVariableOpReadVariableOpconv1d_30/bias*
_output_shapes
:@*
dtype0
?
conv1d_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_31/kernel
y
$conv1d_31/kernel/Read/ReadVariableOpReadVariableOpconv1d_31/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_31/bias
m
"conv1d_31/bias/Read/ReadVariableOpReadVariableOpconv1d_31/bias*
_output_shapes
:@*
dtype0
?
conv1d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_32/kernel
y
$conv1d_32/kernel/Read/ReadVariableOpReadVariableOpconv1d_32/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_32/bias
m
"conv1d_32/bias/Read/ReadVariableOpReadVariableOpconv1d_32/bias*
_output_shapes
:@*
dtype0
?
conv1d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_33/kernel
y
$conv1d_33/kernel/Read/ReadVariableOpReadVariableOpconv1d_33/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_33/bias
m
"conv1d_33/bias/Read/ReadVariableOpReadVariableOpconv1d_33/bias*
_output_shapes
:@*
dtype0
?
conv1d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_34/kernel
y
$conv1d_34/kernel/Read/ReadVariableOpReadVariableOpconv1d_34/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_34/bias
m
"conv1d_34/bias/Read/ReadVariableOpReadVariableOpconv1d_34/bias*
_output_shapes
:@*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	?*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
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
Adam/conv1d_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv1d_28/kernel/m
?
+Adam/conv1d_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_28/kernel/m*"
_output_shapes
:@*
dtype0
?
Adam/conv1d_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_28/bias/m
{
)Adam/conv1d_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_28/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_29/kernel/m
?
+Adam/conv1d_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_29/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_29/bias/m
{
)Adam/conv1d_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_29/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_30/kernel/m
?
+Adam/conv1d_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_30/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_30/bias/m
{
)Adam/conv1d_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_30/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_31/kernel/m
?
+Adam/conv1d_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_31/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_31/bias/m
{
)Adam/conv1d_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_31/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_32/kernel/m
?
+Adam/conv1d_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_32/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_32/bias/m
{
)Adam/conv1d_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_32/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_33/kernel/m
?
+Adam/conv1d_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_33/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_33/bias/m
{
)Adam/conv1d_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_33/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_34/kernel/m
?
+Adam/conv1d_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_34/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_34/bias/m
{
)Adam/conv1d_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_34/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_8/kernel/m
?
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1d_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv1d_28/kernel/v
?
+Adam/conv1d_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_28/kernel/v*"
_output_shapes
:@*
dtype0
?
Adam/conv1d_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_28/bias/v
{
)Adam/conv1d_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_28/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_29/kernel/v
?
+Adam/conv1d_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_29/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_29/bias/v
{
)Adam/conv1d_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_29/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_30/kernel/v
?
+Adam/conv1d_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_30/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_30/bias/v
{
)Adam/conv1d_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_30/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_31/kernel/v
?
+Adam/conv1d_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_31/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_31/bias/v
{
)Adam/conv1d_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_31/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_32/kernel/v
?
+Adam/conv1d_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_32/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_32/bias/v
{
)Adam/conv1d_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_32/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_33/kernel/v
?
+Adam/conv1d_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_33/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_33/bias/v
{
)Adam/conv1d_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_33/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_34/kernel/v
?
+Adam/conv1d_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_34/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_34/bias/v
{
)Adam/conv1d_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_34/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_8/kernel/v
?
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:*
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
 bias
!	variables
"regularization_losses
#trainable_variables
$	keras_api
h

%kernel
&bias
'	variables
(regularization_losses
)trainable_variables
*	keras_api
h

+kernel
,bias
-	variables
.regularization_losses
/trainable_variables
0	keras_api
h

1kernel
2bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
h

7kernel
8bias
9	variables
:regularization_losses
;trainable_variables
<	keras_api
R
=	variables
>regularization_losses
?trainable_variables
@	keras_api
h

Akernel
Bbias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
 
h

Gkernel
Hbias
I	variables
Jregularization_losses
Ktrainable_variables
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
Rmetrics
Snon_trainable_variables

Tlayers
	variables
Ulayer_regularization_losses
regularization_losses
trainable_variables
Vlayer_metrics
 
\Z
VARIABLE_VALUEconv1d_28/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_28/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Wmetrics
Xnon_trainable_variables

Ylayers
Zlayer_regularization_losses
	variables
regularization_losses
trainable_variables
[layer_metrics
\Z
VARIABLE_VALUEconv1d_29/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_29/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
\metrics
]non_trainable_variables

^layers
_layer_regularization_losses
	variables
regularization_losses
trainable_variables
`layer_metrics
\Z
VARIABLE_VALUEconv1d_30/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_30/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1
 

0
 1
?
ametrics
bnon_trainable_variables

clayers
dlayer_regularization_losses
!	variables
"regularization_losses
#trainable_variables
elayer_metrics
\Z
VARIABLE_VALUEconv1d_31/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_31/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
?
fmetrics
gnon_trainable_variables

hlayers
ilayer_regularization_losses
'	variables
(regularization_losses
)trainable_variables
jlayer_metrics
\Z
VARIABLE_VALUEconv1d_32/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_32/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
 

+0
,1
?
kmetrics
lnon_trainable_variables

mlayers
nlayer_regularization_losses
-	variables
.regularization_losses
/trainable_variables
olayer_metrics
\Z
VARIABLE_VALUEconv1d_33/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_33/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
?
pmetrics
qnon_trainable_variables

rlayers
slayer_regularization_losses
3	variables
4regularization_losses
5trainable_variables
tlayer_metrics
\Z
VARIABLE_VALUEconv1d_34/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_34/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81
 

70
81
?
umetrics
vnon_trainable_variables

wlayers
xlayer_regularization_losses
9	variables
:regularization_losses
;trainable_variables
ylayer_metrics
 
 
 
?
zmetrics
{non_trainable_variables

|layers
}layer_regularization_losses
=	variables
>regularization_losses
?trainable_variables
~layer_metrics
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1
 

A0
B1
?
metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
C	variables
Dregularization_losses
Etrainable_variables
?layer_metrics
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

G0
H1
 

G0
H1
?
?metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
I	variables
Jregularization_losses
Ktrainable_variables
?layer_metrics
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

?0
?1
 
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
VARIABLE_VALUEAdam/conv1d_28/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_28/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_29/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_29/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_30/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_30/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_31/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_31/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_32/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_32/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_33/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_33/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_34/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_34/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_28/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_28/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_29/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_29/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_30/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_30/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_31/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_31/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_32/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_32/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_33/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_33/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_34/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_34/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_catserving_default_convconv1d_28/kernelconv1d_28/biasconv1d_29/kernelconv1d_29/biasconv1d_30/kernelconv1d_30/biasconv1d_31/kernelconv1d_31/biasconv1d_32/kernelconv1d_32/biasconv1d_33/kernelconv1d_33/biasconv1d_34/kernelconv1d_34/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_422367
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv1d_28/kernel/Read/ReadVariableOp"conv1d_28/bias/Read/ReadVariableOp$conv1d_29/kernel/Read/ReadVariableOp"conv1d_29/bias/Read/ReadVariableOp$conv1d_30/kernel/Read/ReadVariableOp"conv1d_30/bias/Read/ReadVariableOp$conv1d_31/kernel/Read/ReadVariableOp"conv1d_31/bias/Read/ReadVariableOp$conv1d_32/kernel/Read/ReadVariableOp"conv1d_32/bias/Read/ReadVariableOp$conv1d_33/kernel/Read/ReadVariableOp"conv1d_33/bias/Read/ReadVariableOp$conv1d_34/kernel/Read/ReadVariableOp"conv1d_34/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv1d_28/kernel/m/Read/ReadVariableOp)Adam/conv1d_28/bias/m/Read/ReadVariableOp+Adam/conv1d_29/kernel/m/Read/ReadVariableOp)Adam/conv1d_29/bias/m/Read/ReadVariableOp+Adam/conv1d_30/kernel/m/Read/ReadVariableOp)Adam/conv1d_30/bias/m/Read/ReadVariableOp+Adam/conv1d_31/kernel/m/Read/ReadVariableOp)Adam/conv1d_31/bias/m/Read/ReadVariableOp+Adam/conv1d_32/kernel/m/Read/ReadVariableOp)Adam/conv1d_32/bias/m/Read/ReadVariableOp+Adam/conv1d_33/kernel/m/Read/ReadVariableOp)Adam/conv1d_33/bias/m/Read/ReadVariableOp+Adam/conv1d_34/kernel/m/Read/ReadVariableOp)Adam/conv1d_34/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp+Adam/conv1d_28/kernel/v/Read/ReadVariableOp)Adam/conv1d_28/bias/v/Read/ReadVariableOp+Adam/conv1d_29/kernel/v/Read/ReadVariableOp)Adam/conv1d_29/bias/v/Read/ReadVariableOp+Adam/conv1d_30/kernel/v/Read/ReadVariableOp)Adam/conv1d_30/bias/v/Read/ReadVariableOp+Adam/conv1d_31/kernel/v/Read/ReadVariableOp)Adam/conv1d_31/bias/v/Read/ReadVariableOp+Adam/conv1d_32/kernel/v/Read/ReadVariableOp)Adam/conv1d_32/bias/v/Read/ReadVariableOp+Adam/conv1d_33/kernel/v/Read/ReadVariableOp)Adam/conv1d_33/bias/v/Read/ReadVariableOp+Adam/conv1d_34/kernel/v/Read/ReadVariableOp)Adam/conv1d_34/bias/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOpConst*L
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
__inference__traced_save_423142
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_28/kernelconv1d_28/biasconv1d_29/kernelconv1d_29/biasconv1d_30/kernelconv1d_30/biasconv1d_31/kernelconv1d_31/biasconv1d_32/kernelconv1d_32/biasconv1d_33/kernelconv1d_33/biasconv1d_34/kernelconv1d_34/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv1d_28/kernel/mAdam/conv1d_28/bias/mAdam/conv1d_29/kernel/mAdam/conv1d_29/bias/mAdam/conv1d_30/kernel/mAdam/conv1d_30/bias/mAdam/conv1d_31/kernel/mAdam/conv1d_31/bias/mAdam/conv1d_32/kernel/mAdam/conv1d_32/bias/mAdam/conv1d_33/kernel/mAdam/conv1d_33/bias/mAdam/conv1d_34/kernel/mAdam/conv1d_34/bias/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/conv1d_28/kernel/vAdam/conv1d_28/bias/vAdam/conv1d_29/kernel/vAdam/conv1d_29/bias/vAdam/conv1d_30/kernel/vAdam/conv1d_30/bias/vAdam/conv1d_31/kernel/vAdam/conv1d_31/bias/vAdam/conv1d_32/kernel/vAdam/conv1d_32/bias/vAdam/conv1d_33/kernel/vAdam/conv1d_33/bias/vAdam/conv1d_34/kernel/vAdam/conv1d_34/bias/vAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/v*K
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
"__inference__traced_restore_423341??
?

?
C__inference_dense_9_layer_call_and_return_conditional_losses_422920

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?
?
E__inference_conv1d_30_layer_call_and_return_conditional_losses_421856

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
E__inference_flatten_4_layer_call_and_return_conditional_losses_422014

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
?3
?
C__inference_model_4_layer_call_and_return_conditional_losses_422183

inputs
inputs_1
conv1d_28_422136
conv1d_28_422138
conv1d_29_422141
conv1d_29_422143
conv1d_30_422146
conv1d_30_422148
conv1d_31_422151
conv1d_31_422153
conv1d_32_422156
conv1d_32_422158
conv1d_33_422161
conv1d_33_422163
conv1d_34_422166
conv1d_34_422168
dense_8_422172
dense_8_422174
dense_9_422177
dense_9_422179
identity??!conv1d_28/StatefulPartitionedCall?!conv1d_29/StatefulPartitionedCall?!conv1d_30/StatefulPartitionedCall?!conv1d_31/StatefulPartitionedCall?!conv1d_32/StatefulPartitionedCall?!conv1d_33/StatefulPartitionedCall?!conv1d_34/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
!conv1d_28/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_28_422136conv1d_28_422138*
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
E__inference_conv1d_28_layer_call_and_return_conditional_losses_4217882#
!conv1d_28/StatefulPartitionedCall?
!conv1d_29/StatefulPartitionedCallStatefulPartitionedCall*conv1d_28/StatefulPartitionedCall:output:0conv1d_29_422141conv1d_29_422143*
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
E__inference_conv1d_29_layer_call_and_return_conditional_losses_4218222#
!conv1d_29/StatefulPartitionedCall?
!conv1d_30/StatefulPartitionedCallStatefulPartitionedCall*conv1d_29/StatefulPartitionedCall:output:0conv1d_30_422146conv1d_30_422148*
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
E__inference_conv1d_30_layer_call_and_return_conditional_losses_4218562#
!conv1d_30/StatefulPartitionedCall?
!conv1d_31/StatefulPartitionedCallStatefulPartitionedCall*conv1d_30/StatefulPartitionedCall:output:0conv1d_31_422151conv1d_31_422153*
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
E__inference_conv1d_31_layer_call_and_return_conditional_losses_4218902#
!conv1d_31/StatefulPartitionedCall?
!conv1d_32/StatefulPartitionedCallStatefulPartitionedCall*conv1d_31/StatefulPartitionedCall:output:0conv1d_32_422156conv1d_32_422158*
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
E__inference_conv1d_32_layer_call_and_return_conditional_losses_4219242#
!conv1d_32/StatefulPartitionedCall?
!conv1d_33/StatefulPartitionedCallStatefulPartitionedCall*conv1d_32/StatefulPartitionedCall:output:0conv1d_33_422161conv1d_33_422163*
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
E__inference_conv1d_33_layer_call_and_return_conditional_losses_4219582#
!conv1d_33/StatefulPartitionedCall?
!conv1d_34/StatefulPartitionedCallStatefulPartitionedCall*conv1d_33/StatefulPartitionedCall:output:0conv1d_34_422166conv1d_34_422168*
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
E__inference_conv1d_34_layer_call_and_return_conditional_losses_4219922#
!conv1d_34/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCall*conv1d_34/StatefulPartitionedCall:output:0*
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
E__inference_flatten_4_layer_call_and_return_conditional_losses_4220142
flatten_4/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_8_422172dense_8_422174*
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
GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_4220332!
dense_8/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_422177dense_9_422179*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_4220602!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0"^conv1d_28/StatefulPartitionedCall"^conv1d_29/StatefulPartitionedCall"^conv1d_30/StatefulPartitionedCall"^conv1d_31/StatefulPartitionedCall"^conv1d_32/StatefulPartitionedCall"^conv1d_33/StatefulPartitionedCall"^conv1d_34/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2F
!conv1d_28/StatefulPartitionedCall!conv1d_28/StatefulPartitionedCall2F
!conv1d_29/StatefulPartitionedCall!conv1d_29/StatefulPartitionedCall2F
!conv1d_30/StatefulPartitionedCall!conv1d_30/StatefulPartitionedCall2F
!conv1d_31/StatefulPartitionedCall!conv1d_31/StatefulPartitionedCall2F
!conv1d_32/StatefulPartitionedCall!conv1d_32/StatefulPartitionedCall2F
!conv1d_33/StatefulPartitionedCall!conv1d_33/StatefulPartitionedCall2F
!conv1d_34/StatefulPartitionedCall!conv1d_34/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:S O
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
*__inference_conv1d_29_layer_call_fn_422743

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
E__inference_conv1d_29_layer_call_and_return_conditional_losses_4218222
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
E__inference_conv1d_29_layer_call_and_return_conditional_losses_422734

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
*__inference_conv1d_31_layer_call_fn_422797

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
E__inference_conv1d_31_layer_call_and_return_conditional_losses_4218902
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
C__inference_dense_9_layer_call_and_return_conditional_losses_422060

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?
?
E__inference_conv1d_31_layer_call_and_return_conditional_losses_422788

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
}
(__inference_dense_8_layer_call_fn_422909

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
GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_4220332
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
ƥ
?
C__inference_model_4_layer_call_and_return_conditional_losses_422486
inputs_0
inputs_19
5conv1d_28_conv1d_expanddims_1_readvariableop_resource-
)conv1d_28_biasadd_readvariableop_resource9
5conv1d_29_conv1d_expanddims_1_readvariableop_resource-
)conv1d_29_biasadd_readvariableop_resource9
5conv1d_30_conv1d_expanddims_1_readvariableop_resource-
)conv1d_30_biasadd_readvariableop_resource9
5conv1d_31_conv1d_expanddims_1_readvariableop_resource-
)conv1d_31_biasadd_readvariableop_resource9
5conv1d_32_conv1d_expanddims_1_readvariableop_resource-
)conv1d_32_biasadd_readvariableop_resource9
5conv1d_33_conv1d_expanddims_1_readvariableop_resource-
)conv1d_33_biasadd_readvariableop_resource9
5conv1d_34_conv1d_expanddims_1_readvariableop_resource-
)conv1d_34_biasadd_readvariableop_resource-
)dense_8_mlcmatmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource-
)dense_9_mlcmatmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identity?? conv1d_28/BiasAdd/ReadVariableOp?,conv1d_28/conv1d/ExpandDims_1/ReadVariableOp? conv1d_29/BiasAdd/ReadVariableOp?,conv1d_29/conv1d/ExpandDims_1/ReadVariableOp? conv1d_30/BiasAdd/ReadVariableOp?,conv1d_30/conv1d/ExpandDims_1/ReadVariableOp? conv1d_31/BiasAdd/ReadVariableOp?,conv1d_31/conv1d/ExpandDims_1/ReadVariableOp? conv1d_32/BiasAdd/ReadVariableOp?,conv1d_32/conv1d/ExpandDims_1/ReadVariableOp? conv1d_33/BiasAdd/ReadVariableOp?,conv1d_33/conv1d/ExpandDims_1/ReadVariableOp? conv1d_34/BiasAdd/ReadVariableOp?,conv1d_34/conv1d/ExpandDims_1/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp? dense_8/MLCMatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp? dense_9/MLCMatMul/ReadVariableOp?
conv1d_28/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_28/Pad/paddings?
conv1d_28/PadPadinputs_0conv1d_28/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
conv1d_28/Pad?
conv1d_28/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_28/conv1d/ExpandDims/dim?
conv1d_28/conv1d/ExpandDims
ExpandDimsconv1d_28/Pad:output:0(conv1d_28/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_28/conv1d/ExpandDims?
,conv1d_28/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_28_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_28/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_28/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_28/conv1d/ExpandDims_1/dim?
conv1d_28/conv1d/ExpandDims_1
ExpandDims4conv1d_28/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_28/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_28/conv1d/ExpandDims_1?
conv1d_28/conv1d	MLCConv2D$conv1d_28/conv1d/ExpandDims:output:0&conv1d_28/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_28/conv1d?
conv1d_28/conv1d/SqueezeSqueezeconv1d_28/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_28/conv1d/Squeeze?
 conv1d_28/BiasAdd/ReadVariableOpReadVariableOp)conv1d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_28/BiasAdd/ReadVariableOp?
conv1d_28/BiasAddBiasAdd!conv1d_28/conv1d/Squeeze:output:0(conv1d_28/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_28/BiasAddz
conv1d_28/ReluReluconv1d_28/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_28/Relu?
conv1d_29/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_29/Pad/paddings?
conv1d_29/PadPadconv1d_28/Relu:activations:0conv1d_29/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_29/Pad?
conv1d_29/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_29/conv1d/ExpandDims/dim?
conv1d_29/conv1d/ExpandDims
ExpandDimsconv1d_29/Pad:output:0(conv1d_29/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_29/conv1d/ExpandDims?
,conv1d_29/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_29_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_29/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_29/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_29/conv1d/ExpandDims_1/dim?
conv1d_29/conv1d/ExpandDims_1
ExpandDims4conv1d_29/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_29/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_29/conv1d/ExpandDims_1?
conv1d_29/conv1d	MLCConv2D$conv1d_29/conv1d/ExpandDims:output:0&conv1d_29/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_29/conv1d?
conv1d_29/conv1d/SqueezeSqueezeconv1d_29/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_29/conv1d/Squeeze?
 conv1d_29/BiasAdd/ReadVariableOpReadVariableOp)conv1d_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_29/BiasAdd/ReadVariableOp?
conv1d_29/BiasAddBiasAdd!conv1d_29/conv1d/Squeeze:output:0(conv1d_29/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_29/BiasAddz
conv1d_29/ReluReluconv1d_29/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_29/Relu?
conv1d_30/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_30/Pad/paddings?
conv1d_30/PadPadconv1d_29/Relu:activations:0conv1d_30/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_30/Pad?
conv1d_30/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_30/conv1d/ExpandDims/dim?
conv1d_30/conv1d/ExpandDims
ExpandDimsconv1d_30/Pad:output:0(conv1d_30/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_30/conv1d/ExpandDims?
,conv1d_30/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_30_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_30/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_30/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_30/conv1d/ExpandDims_1/dim?
conv1d_30/conv1d/ExpandDims_1
ExpandDims4conv1d_30/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_30/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_30/conv1d/ExpandDims_1?
conv1d_30/conv1d	MLCConv2D$conv1d_30/conv1d/ExpandDims:output:0&conv1d_30/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_30/conv1d?
conv1d_30/conv1d/SqueezeSqueezeconv1d_30/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_30/conv1d/Squeeze?
 conv1d_30/BiasAdd/ReadVariableOpReadVariableOp)conv1d_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_30/BiasAdd/ReadVariableOp?
conv1d_30/BiasAddBiasAdd!conv1d_30/conv1d/Squeeze:output:0(conv1d_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_30/BiasAddz
conv1d_30/ReluReluconv1d_30/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_30/Relu?
conv1d_31/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_31/Pad/paddings?
conv1d_31/PadPadconv1d_30/Relu:activations:0conv1d_31/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_31/Pad?
conv1d_31/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_31/conv1d/ExpandDims/dim?
conv1d_31/conv1d/ExpandDims
ExpandDimsconv1d_31/Pad:output:0(conv1d_31/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_31/conv1d/ExpandDims?
,conv1d_31/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_31_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_31/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_31/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_31/conv1d/ExpandDims_1/dim?
conv1d_31/conv1d/ExpandDims_1
ExpandDims4conv1d_31/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_31/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_31/conv1d/ExpandDims_1?
conv1d_31/conv1d	MLCConv2D$conv1d_31/conv1d/ExpandDims:output:0&conv1d_31/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_31/conv1d?
conv1d_31/conv1d/SqueezeSqueezeconv1d_31/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_31/conv1d/Squeeze?
 conv1d_31/BiasAdd/ReadVariableOpReadVariableOp)conv1d_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_31/BiasAdd/ReadVariableOp?
conv1d_31/BiasAddBiasAdd!conv1d_31/conv1d/Squeeze:output:0(conv1d_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_31/BiasAddz
conv1d_31/ReluReluconv1d_31/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_31/Relu?
conv1d_32/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_32/Pad/paddings?
conv1d_32/PadPadconv1d_31/Relu:activations:0conv1d_32/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_32/Pad?
conv1d_32/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_32/conv1d/ExpandDims/dim?
conv1d_32/conv1d/ExpandDims
ExpandDimsconv1d_32/Pad:output:0(conv1d_32/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_32/conv1d/ExpandDims?
,conv1d_32/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_32_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_32/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_32/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_32/conv1d/ExpandDims_1/dim?
conv1d_32/conv1d/ExpandDims_1
ExpandDims4conv1d_32/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_32/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_32/conv1d/ExpandDims_1?
conv1d_32/conv1d	MLCConv2D$conv1d_32/conv1d/ExpandDims:output:0&conv1d_32/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_32/conv1d?
conv1d_32/conv1d/SqueezeSqueezeconv1d_32/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_32/conv1d/Squeeze?
 conv1d_32/BiasAdd/ReadVariableOpReadVariableOp)conv1d_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_32/BiasAdd/ReadVariableOp?
conv1d_32/BiasAddBiasAdd!conv1d_32/conv1d/Squeeze:output:0(conv1d_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_32/BiasAddz
conv1d_32/ReluReluconv1d_32/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_32/Relu?
conv1d_33/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_33/Pad/paddings?
conv1d_33/PadPadconv1d_32/Relu:activations:0conv1d_33/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_33/Pad?
conv1d_33/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_33/conv1d/ExpandDims/dim?
conv1d_33/conv1d/ExpandDims
ExpandDimsconv1d_33/Pad:output:0(conv1d_33/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_33/conv1d/ExpandDims?
,conv1d_33/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_33_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_33/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_33/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_33/conv1d/ExpandDims_1/dim?
conv1d_33/conv1d/ExpandDims_1
ExpandDims4conv1d_33/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_33/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_33/conv1d/ExpandDims_1?
conv1d_33/conv1d	MLCConv2D$conv1d_33/conv1d/ExpandDims:output:0&conv1d_33/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_33/conv1d?
conv1d_33/conv1d/SqueezeSqueezeconv1d_33/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_33/conv1d/Squeeze?
 conv1d_33/BiasAdd/ReadVariableOpReadVariableOp)conv1d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_33/BiasAdd/ReadVariableOp?
conv1d_33/BiasAddBiasAdd!conv1d_33/conv1d/Squeeze:output:0(conv1d_33/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_33/BiasAddz
conv1d_33/ReluReluconv1d_33/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_33/Relu?
conv1d_34/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_34/Pad/paddings?
conv1d_34/PadPadconv1d_33/Relu:activations:0conv1d_34/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_34/Pad?
conv1d_34/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_34/conv1d/ExpandDims/dim?
conv1d_34/conv1d/ExpandDims
ExpandDimsconv1d_34/Pad:output:0(conv1d_34/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_34/conv1d/ExpandDims?
,conv1d_34/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_34_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_34/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_34/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_34/conv1d/ExpandDims_1/dim?
conv1d_34/conv1d/ExpandDims_1
ExpandDims4conv1d_34/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_34/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_34/conv1d/ExpandDims_1?
conv1d_34/conv1d	MLCConv2D$conv1d_34/conv1d/ExpandDims:output:0&conv1d_34/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_34/conv1d?
conv1d_34/conv1d/SqueezeSqueezeconv1d_34/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_34/conv1d/Squeeze?
 conv1d_34/BiasAdd/ReadVariableOpReadVariableOp)conv1d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_34/BiasAdd/ReadVariableOp?
conv1d_34/BiasAddBiasAdd!conv1d_34/conv1d/Squeeze:output:0(conv1d_34/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_34/BiasAddz
conv1d_34/ReluReluconv1d_34/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_34/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_4/Const?
flatten_4/ReshapeReshapeconv1d_34/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
 dense_8/MLCMatMul/ReadVariableOpReadVariableOp)dense_8_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_8/MLCMatMul/ReadVariableOp?
dense_8/MLCMatMul	MLCMatMulflatten_4/Reshape:output:0(dense_8/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MLCMatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MLCMatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAddp
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_8/Relu?
 dense_9/MLCMatMul/ReadVariableOpReadVariableOp)dense_9_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_9/MLCMatMul/ReadVariableOp?
dense_9/MLCMatMul	MLCMatMuldense_8/Relu:activations:0(dense_9/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MLCMatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MLCMatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAddy
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_9/Softmax?
IdentityIdentitydense_9/Softmax:softmax:0!^conv1d_28/BiasAdd/ReadVariableOp-^conv1d_28/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_29/BiasAdd/ReadVariableOp-^conv1d_29/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_30/BiasAdd/ReadVariableOp-^conv1d_30/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_31/BiasAdd/ReadVariableOp-^conv1d_31/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_32/BiasAdd/ReadVariableOp-^conv1d_32/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_33/BiasAdd/ReadVariableOp-^conv1d_33/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_34/BiasAdd/ReadVariableOp-^conv1d_34/conv1d/ExpandDims_1/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/MLCMatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2D
 conv1d_28/BiasAdd/ReadVariableOp conv1d_28/BiasAdd/ReadVariableOp2\
,conv1d_28/conv1d/ExpandDims_1/ReadVariableOp,conv1d_28/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_29/BiasAdd/ReadVariableOp conv1d_29/BiasAdd/ReadVariableOp2\
,conv1d_29/conv1d/ExpandDims_1/ReadVariableOp,conv1d_29/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_30/BiasAdd/ReadVariableOp conv1d_30/BiasAdd/ReadVariableOp2\
,conv1d_30/conv1d/ExpandDims_1/ReadVariableOp,conv1d_30/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_31/BiasAdd/ReadVariableOp conv1d_31/BiasAdd/ReadVariableOp2\
,conv1d_31/conv1d/ExpandDims_1/ReadVariableOp,conv1d_31/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_32/BiasAdd/ReadVariableOp conv1d_32/BiasAdd/ReadVariableOp2\
,conv1d_32/conv1d/ExpandDims_1/ReadVariableOp,conv1d_32/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_33/BiasAdd/ReadVariableOp conv1d_33/BiasAdd/ReadVariableOp2\
,conv1d_33/conv1d/ExpandDims_1/ReadVariableOp,conv1d_33/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_34/BiasAdd/ReadVariableOp conv1d_34/BiasAdd/ReadVariableOp2\
,conv1d_34/conv1d/ExpandDims_1/ReadVariableOp,conv1d_34/conv1d/ExpandDims_1/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/MLCMatMul/ReadVariableOp dense_8/MLCMatMul/ReadVariableOp2@
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
:?????????
"
_user_specified_name
inputs/1
?
?
E__inference_conv1d_33_layer_call_and_return_conditional_losses_422842

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
C__inference_dense_8_layer_call_and_return_conditional_losses_422900

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
?
?
E__inference_conv1d_32_layer_call_and_return_conditional_losses_421924

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
*__inference_conv1d_28_layer_call_fn_422716

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
E__inference_conv1d_28_layer_call_and_return_conditional_losses_4217882
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
?3
?
C__inference_model_4_layer_call_and_return_conditional_losses_422077
conv
cat
conv1d_28_421799
conv1d_28_421801
conv1d_29_421833
conv1d_29_421835
conv1d_30_421867
conv1d_30_421869
conv1d_31_421901
conv1d_31_421903
conv1d_32_421935
conv1d_32_421937
conv1d_33_421969
conv1d_33_421971
conv1d_34_422003
conv1d_34_422005
dense_8_422044
dense_8_422046
dense_9_422071
dense_9_422073
identity??!conv1d_28/StatefulPartitionedCall?!conv1d_29/StatefulPartitionedCall?!conv1d_30/StatefulPartitionedCall?!conv1d_31/StatefulPartitionedCall?!conv1d_32/StatefulPartitionedCall?!conv1d_33/StatefulPartitionedCall?!conv1d_34/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
!conv1d_28/StatefulPartitionedCallStatefulPartitionedCallconvconv1d_28_421799conv1d_28_421801*
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
E__inference_conv1d_28_layer_call_and_return_conditional_losses_4217882#
!conv1d_28/StatefulPartitionedCall?
!conv1d_29/StatefulPartitionedCallStatefulPartitionedCall*conv1d_28/StatefulPartitionedCall:output:0conv1d_29_421833conv1d_29_421835*
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
E__inference_conv1d_29_layer_call_and_return_conditional_losses_4218222#
!conv1d_29/StatefulPartitionedCall?
!conv1d_30/StatefulPartitionedCallStatefulPartitionedCall*conv1d_29/StatefulPartitionedCall:output:0conv1d_30_421867conv1d_30_421869*
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
E__inference_conv1d_30_layer_call_and_return_conditional_losses_4218562#
!conv1d_30/StatefulPartitionedCall?
!conv1d_31/StatefulPartitionedCallStatefulPartitionedCall*conv1d_30/StatefulPartitionedCall:output:0conv1d_31_421901conv1d_31_421903*
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
E__inference_conv1d_31_layer_call_and_return_conditional_losses_4218902#
!conv1d_31/StatefulPartitionedCall?
!conv1d_32/StatefulPartitionedCallStatefulPartitionedCall*conv1d_31/StatefulPartitionedCall:output:0conv1d_32_421935conv1d_32_421937*
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
E__inference_conv1d_32_layer_call_and_return_conditional_losses_4219242#
!conv1d_32/StatefulPartitionedCall?
!conv1d_33/StatefulPartitionedCallStatefulPartitionedCall*conv1d_32/StatefulPartitionedCall:output:0conv1d_33_421969conv1d_33_421971*
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
E__inference_conv1d_33_layer_call_and_return_conditional_losses_4219582#
!conv1d_33/StatefulPartitionedCall?
!conv1d_34/StatefulPartitionedCallStatefulPartitionedCall*conv1d_33/StatefulPartitionedCall:output:0conv1d_34_422003conv1d_34_422005*
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
E__inference_conv1d_34_layer_call_and_return_conditional_losses_4219922#
!conv1d_34/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCall*conv1d_34/StatefulPartitionedCall:output:0*
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
E__inference_flatten_4_layer_call_and_return_conditional_losses_4220142
flatten_4/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_8_422044dense_8_422046*
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
GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_4220332!
dense_8/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_422071dense_9_422073*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_4220602!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0"^conv1d_28/StatefulPartitionedCall"^conv1d_29/StatefulPartitionedCall"^conv1d_30/StatefulPartitionedCall"^conv1d_31/StatefulPartitionedCall"^conv1d_32/StatefulPartitionedCall"^conv1d_33/StatefulPartitionedCall"^conv1d_34/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2F
!conv1d_28/StatefulPartitionedCall!conv1d_28/StatefulPartitionedCall2F
!conv1d_29/StatefulPartitionedCall!conv1d_29/StatefulPartitionedCall2F
!conv1d_30/StatefulPartitionedCall!conv1d_30/StatefulPartitionedCall2F
!conv1d_31/StatefulPartitionedCall!conv1d_31/StatefulPartitionedCall2F
!conv1d_32/StatefulPartitionedCall!conv1d_32/StatefulPartitionedCall2F
!conv1d_33/StatefulPartitionedCall!conv1d_33/StatefulPartitionedCall2F
!conv1d_34/StatefulPartitionedCall!conv1d_34/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
?
?
E__inference_conv1d_28_layer_call_and_return_conditional_losses_421788

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
E__inference_conv1d_32_layer_call_and_return_conditional_losses_422815

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
*__inference_conv1d_30_layer_call_fn_422770

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
E__inference_conv1d_30_layer_call_and_return_conditional_losses_4218562
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
*__inference_conv1d_34_layer_call_fn_422878

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
E__inference_conv1d_34_layer_call_and_return_conditional_losses_4219922
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
*__inference_conv1d_33_layer_call_fn_422851

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
E__inference_conv1d_33_layer_call_and_return_conditional_losses_4219582
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
?
(__inference_model_4_layer_call_fn_422647
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
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_4221832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
?
?
E__inference_conv1d_34_layer_call_and_return_conditional_losses_422869

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
?|
?
__inference__traced_save_423142
file_prefix/
+savev2_conv1d_28_kernel_read_readvariableop-
)savev2_conv1d_28_bias_read_readvariableop/
+savev2_conv1d_29_kernel_read_readvariableop-
)savev2_conv1d_29_bias_read_readvariableop/
+savev2_conv1d_30_kernel_read_readvariableop-
)savev2_conv1d_30_bias_read_readvariableop/
+savev2_conv1d_31_kernel_read_readvariableop-
)savev2_conv1d_31_bias_read_readvariableop/
+savev2_conv1d_32_kernel_read_readvariableop-
)savev2_conv1d_32_bias_read_readvariableop/
+savev2_conv1d_33_kernel_read_readvariableop-
)savev2_conv1d_33_bias_read_readvariableop/
+savev2_conv1d_34_kernel_read_readvariableop-
)savev2_conv1d_34_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv1d_28_kernel_m_read_readvariableop4
0savev2_adam_conv1d_28_bias_m_read_readvariableop6
2savev2_adam_conv1d_29_kernel_m_read_readvariableop4
0savev2_adam_conv1d_29_bias_m_read_readvariableop6
2savev2_adam_conv1d_30_kernel_m_read_readvariableop4
0savev2_adam_conv1d_30_bias_m_read_readvariableop6
2savev2_adam_conv1d_31_kernel_m_read_readvariableop4
0savev2_adam_conv1d_31_bias_m_read_readvariableop6
2savev2_adam_conv1d_32_kernel_m_read_readvariableop4
0savev2_adam_conv1d_32_bias_m_read_readvariableop6
2savev2_adam_conv1d_33_kernel_m_read_readvariableop4
0savev2_adam_conv1d_33_bias_m_read_readvariableop6
2savev2_adam_conv1d_34_kernel_m_read_readvariableop4
0savev2_adam_conv1d_34_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop6
2savev2_adam_conv1d_28_kernel_v_read_readvariableop4
0savev2_adam_conv1d_28_bias_v_read_readvariableop6
2savev2_adam_conv1d_29_kernel_v_read_readvariableop4
0savev2_adam_conv1d_29_bias_v_read_readvariableop6
2savev2_adam_conv1d_30_kernel_v_read_readvariableop4
0savev2_adam_conv1d_30_bias_v_read_readvariableop6
2savev2_adam_conv1d_31_kernel_v_read_readvariableop4
0savev2_adam_conv1d_31_bias_v_read_readvariableop6
2savev2_adam_conv1d_32_kernel_v_read_readvariableop4
0savev2_adam_conv1d_32_bias_v_read_readvariableop6
2savev2_adam_conv1d_33_kernel_v_read_readvariableop4
0savev2_adam_conv1d_33_bias_v_read_readvariableop6
2savev2_adam_conv1d_34_kernel_v_read_readvariableop4
0savev2_adam_conv1d_34_bias_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop
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
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv1d_28_kernel_read_readvariableop)savev2_conv1d_28_bias_read_readvariableop+savev2_conv1d_29_kernel_read_readvariableop)savev2_conv1d_29_bias_read_readvariableop+savev2_conv1d_30_kernel_read_readvariableop)savev2_conv1d_30_bias_read_readvariableop+savev2_conv1d_31_kernel_read_readvariableop)savev2_conv1d_31_bias_read_readvariableop+savev2_conv1d_32_kernel_read_readvariableop)savev2_conv1d_32_bias_read_readvariableop+savev2_conv1d_33_kernel_read_readvariableop)savev2_conv1d_33_bias_read_readvariableop+savev2_conv1d_34_kernel_read_readvariableop)savev2_conv1d_34_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv1d_28_kernel_m_read_readvariableop0savev2_adam_conv1d_28_bias_m_read_readvariableop2savev2_adam_conv1d_29_kernel_m_read_readvariableop0savev2_adam_conv1d_29_bias_m_read_readvariableop2savev2_adam_conv1d_30_kernel_m_read_readvariableop0savev2_adam_conv1d_30_bias_m_read_readvariableop2savev2_adam_conv1d_31_kernel_m_read_readvariableop0savev2_adam_conv1d_31_bias_m_read_readvariableop2savev2_adam_conv1d_32_kernel_m_read_readvariableop0savev2_adam_conv1d_32_bias_m_read_readvariableop2savev2_adam_conv1d_33_kernel_m_read_readvariableop0savev2_adam_conv1d_33_bias_m_read_readvariableop2savev2_adam_conv1d_34_kernel_m_read_readvariableop0savev2_adam_conv1d_34_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop2savev2_adam_conv1d_28_kernel_v_read_readvariableop0savev2_adam_conv1d_28_bias_v_read_readvariableop2savev2_adam_conv1d_29_kernel_v_read_readvariableop0savev2_adam_conv1d_29_bias_v_read_readvariableop2savev2_adam_conv1d_30_kernel_v_read_readvariableop0savev2_adam_conv1d_30_bias_v_read_readvariableop2savev2_adam_conv1d_31_kernel_v_read_readvariableop0savev2_adam_conv1d_31_bias_v_read_readvariableop2savev2_adam_conv1d_32_kernel_v_read_readvariableop0savev2_adam_conv1d_32_bias_v_read_readvariableop2savev2_adam_conv1d_33_kernel_v_read_readvariableop0savev2_adam_conv1d_33_bias_v_read_readvariableop2savev2_adam_conv1d_34_kernel_v_read_readvariableop0savev2_adam_conv1d_34_bias_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:	?:::: : : : : : : : : :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:	?::::@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:	?:::: 2(
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
:	?: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::
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
:	?: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::(.$
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
:	?: =

_output_shapes
::$> 

_output_shapes

:: ?

_output_shapes
::@

_output_shapes
: 
?
?
E__inference_conv1d_29_layer_call_and_return_conditional_losses_421822

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
C__inference_model_4_layer_call_and_return_conditional_losses_422276

inputs
inputs_1
conv1d_28_422229
conv1d_28_422231
conv1d_29_422234
conv1d_29_422236
conv1d_30_422239
conv1d_30_422241
conv1d_31_422244
conv1d_31_422246
conv1d_32_422249
conv1d_32_422251
conv1d_33_422254
conv1d_33_422256
conv1d_34_422259
conv1d_34_422261
dense_8_422265
dense_8_422267
dense_9_422270
dense_9_422272
identity??!conv1d_28/StatefulPartitionedCall?!conv1d_29/StatefulPartitionedCall?!conv1d_30/StatefulPartitionedCall?!conv1d_31/StatefulPartitionedCall?!conv1d_32/StatefulPartitionedCall?!conv1d_33/StatefulPartitionedCall?!conv1d_34/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
!conv1d_28/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_28_422229conv1d_28_422231*
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
E__inference_conv1d_28_layer_call_and_return_conditional_losses_4217882#
!conv1d_28/StatefulPartitionedCall?
!conv1d_29/StatefulPartitionedCallStatefulPartitionedCall*conv1d_28/StatefulPartitionedCall:output:0conv1d_29_422234conv1d_29_422236*
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
E__inference_conv1d_29_layer_call_and_return_conditional_losses_4218222#
!conv1d_29/StatefulPartitionedCall?
!conv1d_30/StatefulPartitionedCallStatefulPartitionedCall*conv1d_29/StatefulPartitionedCall:output:0conv1d_30_422239conv1d_30_422241*
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
E__inference_conv1d_30_layer_call_and_return_conditional_losses_4218562#
!conv1d_30/StatefulPartitionedCall?
!conv1d_31/StatefulPartitionedCallStatefulPartitionedCall*conv1d_30/StatefulPartitionedCall:output:0conv1d_31_422244conv1d_31_422246*
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
E__inference_conv1d_31_layer_call_and_return_conditional_losses_4218902#
!conv1d_31/StatefulPartitionedCall?
!conv1d_32/StatefulPartitionedCallStatefulPartitionedCall*conv1d_31/StatefulPartitionedCall:output:0conv1d_32_422249conv1d_32_422251*
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
E__inference_conv1d_32_layer_call_and_return_conditional_losses_4219242#
!conv1d_32/StatefulPartitionedCall?
!conv1d_33/StatefulPartitionedCallStatefulPartitionedCall*conv1d_32/StatefulPartitionedCall:output:0conv1d_33_422254conv1d_33_422256*
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
E__inference_conv1d_33_layer_call_and_return_conditional_losses_4219582#
!conv1d_33/StatefulPartitionedCall?
!conv1d_34/StatefulPartitionedCallStatefulPartitionedCall*conv1d_33/StatefulPartitionedCall:output:0conv1d_34_422259conv1d_34_422261*
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
E__inference_conv1d_34_layer_call_and_return_conditional_losses_4219922#
!conv1d_34/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCall*conv1d_34/StatefulPartitionedCall:output:0*
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
E__inference_flatten_4_layer_call_and_return_conditional_losses_4220142
flatten_4/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_8_422265dense_8_422267*
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
GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_4220332!
dense_8/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_422270dense_9_422272*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_4220602!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0"^conv1d_28/StatefulPartitionedCall"^conv1d_29/StatefulPartitionedCall"^conv1d_30/StatefulPartitionedCall"^conv1d_31/StatefulPartitionedCall"^conv1d_32/StatefulPartitionedCall"^conv1d_33/StatefulPartitionedCall"^conv1d_34/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2F
!conv1d_28/StatefulPartitionedCall!conv1d_28/StatefulPartitionedCall2F
!conv1d_29/StatefulPartitionedCall!conv1d_29/StatefulPartitionedCall2F
!conv1d_30/StatefulPartitionedCall!conv1d_30/StatefulPartitionedCall2F
!conv1d_31/StatefulPartitionedCall!conv1d_31/StatefulPartitionedCall2F
!conv1d_32/StatefulPartitionedCall!conv1d_32/StatefulPartitionedCall2F
!conv1d_33/StatefulPartitionedCall!conv1d_33/StatefulPartitionedCall2F
!conv1d_34/StatefulPartitionedCall!conv1d_34/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
? 
"__inference__traced_restore_423341
file_prefix%
!assignvariableop_conv1d_28_kernel%
!assignvariableop_1_conv1d_28_bias'
#assignvariableop_2_conv1d_29_kernel%
!assignvariableop_3_conv1d_29_bias'
#assignvariableop_4_conv1d_30_kernel%
!assignvariableop_5_conv1d_30_bias'
#assignvariableop_6_conv1d_31_kernel%
!assignvariableop_7_conv1d_31_bias'
#assignvariableop_8_conv1d_32_kernel%
!assignvariableop_9_conv1d_32_bias(
$assignvariableop_10_conv1d_33_kernel&
"assignvariableop_11_conv1d_33_bias(
$assignvariableop_12_conv1d_34_kernel&
"assignvariableop_13_conv1d_34_bias&
"assignvariableop_14_dense_8_kernel$
 assignvariableop_15_dense_8_bias&
"assignvariableop_16_dense_9_kernel$
 assignvariableop_17_dense_9_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate
assignvariableop_23_total
assignvariableop_24_count
assignvariableop_25_total_1
assignvariableop_26_count_1/
+assignvariableop_27_adam_conv1d_28_kernel_m-
)assignvariableop_28_adam_conv1d_28_bias_m/
+assignvariableop_29_adam_conv1d_29_kernel_m-
)assignvariableop_30_adam_conv1d_29_bias_m/
+assignvariableop_31_adam_conv1d_30_kernel_m-
)assignvariableop_32_adam_conv1d_30_bias_m/
+assignvariableop_33_adam_conv1d_31_kernel_m-
)assignvariableop_34_adam_conv1d_31_bias_m/
+assignvariableop_35_adam_conv1d_32_kernel_m-
)assignvariableop_36_adam_conv1d_32_bias_m/
+assignvariableop_37_adam_conv1d_33_kernel_m-
)assignvariableop_38_adam_conv1d_33_bias_m/
+assignvariableop_39_adam_conv1d_34_kernel_m-
)assignvariableop_40_adam_conv1d_34_bias_m-
)assignvariableop_41_adam_dense_8_kernel_m+
'assignvariableop_42_adam_dense_8_bias_m-
)assignvariableop_43_adam_dense_9_kernel_m+
'assignvariableop_44_adam_dense_9_bias_m/
+assignvariableop_45_adam_conv1d_28_kernel_v-
)assignvariableop_46_adam_conv1d_28_bias_v/
+assignvariableop_47_adam_conv1d_29_kernel_v-
)assignvariableop_48_adam_conv1d_29_bias_v/
+assignvariableop_49_adam_conv1d_30_kernel_v-
)assignvariableop_50_adam_conv1d_30_bias_v/
+assignvariableop_51_adam_conv1d_31_kernel_v-
)assignvariableop_52_adam_conv1d_31_bias_v/
+assignvariableop_53_adam_conv1d_32_kernel_v-
)assignvariableop_54_adam_conv1d_32_bias_v/
+assignvariableop_55_adam_conv1d_33_kernel_v-
)assignvariableop_56_adam_conv1d_33_bias_v/
+assignvariableop_57_adam_conv1d_34_kernel_v-
)assignvariableop_58_adam_conv1d_34_bias_v-
)assignvariableop_59_adam_dense_8_kernel_v+
'assignvariableop_60_adam_dense_8_bias_v-
)assignvariableop_61_adam_dense_9_kernel_v+
'assignvariableop_62_adam_dense_9_bias_v
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
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_28_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_28_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv1d_29_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv1d_29_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv1d_30_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv1d_30_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv1d_31_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv1d_31_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv1d_32_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv1d_32_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv1d_33_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv1d_33_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv1d_34_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv1d_34_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_8_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_8_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_9_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_9_biasIdentity_17:output:0"/device:CPU:0*
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
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv1d_28_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv1d_28_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv1d_29_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv1d_29_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv1d_30_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv1d_30_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv1d_31_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv1d_31_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv1d_32_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv1d_32_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv1d_33_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv1d_33_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv1d_34_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv1d_34_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_8_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_8_bias_mIdentity_42:output:0"/device:CPU:0*
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
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv1d_28_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv1d_28_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv1d_29_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv1d_29_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv1d_30_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv1d_30_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv1d_31_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv1d_31_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv1d_32_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv1d_32_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv1d_33_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv1d_33_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv1d_34_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv1d_34_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_8_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_dense_8_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_dense_9_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_dense_9_bias_vIdentity_62:output:0"/device:CPU:0*
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
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_422884

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
E__inference_conv1d_31_layer_call_and_return_conditional_losses_421890

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
C__inference_dense_8_layer_call_and_return_conditional_losses_422033

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
ƥ
?
C__inference_model_4_layer_call_and_return_conditional_losses_422605
inputs_0
inputs_19
5conv1d_28_conv1d_expanddims_1_readvariableop_resource-
)conv1d_28_biasadd_readvariableop_resource9
5conv1d_29_conv1d_expanddims_1_readvariableop_resource-
)conv1d_29_biasadd_readvariableop_resource9
5conv1d_30_conv1d_expanddims_1_readvariableop_resource-
)conv1d_30_biasadd_readvariableop_resource9
5conv1d_31_conv1d_expanddims_1_readvariableop_resource-
)conv1d_31_biasadd_readvariableop_resource9
5conv1d_32_conv1d_expanddims_1_readvariableop_resource-
)conv1d_32_biasadd_readvariableop_resource9
5conv1d_33_conv1d_expanddims_1_readvariableop_resource-
)conv1d_33_biasadd_readvariableop_resource9
5conv1d_34_conv1d_expanddims_1_readvariableop_resource-
)conv1d_34_biasadd_readvariableop_resource-
)dense_8_mlcmatmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource-
)dense_9_mlcmatmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identity?? conv1d_28/BiasAdd/ReadVariableOp?,conv1d_28/conv1d/ExpandDims_1/ReadVariableOp? conv1d_29/BiasAdd/ReadVariableOp?,conv1d_29/conv1d/ExpandDims_1/ReadVariableOp? conv1d_30/BiasAdd/ReadVariableOp?,conv1d_30/conv1d/ExpandDims_1/ReadVariableOp? conv1d_31/BiasAdd/ReadVariableOp?,conv1d_31/conv1d/ExpandDims_1/ReadVariableOp? conv1d_32/BiasAdd/ReadVariableOp?,conv1d_32/conv1d/ExpandDims_1/ReadVariableOp? conv1d_33/BiasAdd/ReadVariableOp?,conv1d_33/conv1d/ExpandDims_1/ReadVariableOp? conv1d_34/BiasAdd/ReadVariableOp?,conv1d_34/conv1d/ExpandDims_1/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp? dense_8/MLCMatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp? dense_9/MLCMatMul/ReadVariableOp?
conv1d_28/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_28/Pad/paddings?
conv1d_28/PadPadinputs_0conv1d_28/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
conv1d_28/Pad?
conv1d_28/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_28/conv1d/ExpandDims/dim?
conv1d_28/conv1d/ExpandDims
ExpandDimsconv1d_28/Pad:output:0(conv1d_28/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_28/conv1d/ExpandDims?
,conv1d_28/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_28_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_28/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_28/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_28/conv1d/ExpandDims_1/dim?
conv1d_28/conv1d/ExpandDims_1
ExpandDims4conv1d_28/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_28/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_28/conv1d/ExpandDims_1?
conv1d_28/conv1d	MLCConv2D$conv1d_28/conv1d/ExpandDims:output:0&conv1d_28/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_28/conv1d?
conv1d_28/conv1d/SqueezeSqueezeconv1d_28/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_28/conv1d/Squeeze?
 conv1d_28/BiasAdd/ReadVariableOpReadVariableOp)conv1d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_28/BiasAdd/ReadVariableOp?
conv1d_28/BiasAddBiasAdd!conv1d_28/conv1d/Squeeze:output:0(conv1d_28/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_28/BiasAddz
conv1d_28/ReluReluconv1d_28/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_28/Relu?
conv1d_29/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_29/Pad/paddings?
conv1d_29/PadPadconv1d_28/Relu:activations:0conv1d_29/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_29/Pad?
conv1d_29/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_29/conv1d/ExpandDims/dim?
conv1d_29/conv1d/ExpandDims
ExpandDimsconv1d_29/Pad:output:0(conv1d_29/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_29/conv1d/ExpandDims?
,conv1d_29/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_29_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_29/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_29/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_29/conv1d/ExpandDims_1/dim?
conv1d_29/conv1d/ExpandDims_1
ExpandDims4conv1d_29/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_29/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_29/conv1d/ExpandDims_1?
conv1d_29/conv1d	MLCConv2D$conv1d_29/conv1d/ExpandDims:output:0&conv1d_29/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_29/conv1d?
conv1d_29/conv1d/SqueezeSqueezeconv1d_29/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_29/conv1d/Squeeze?
 conv1d_29/BiasAdd/ReadVariableOpReadVariableOp)conv1d_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_29/BiasAdd/ReadVariableOp?
conv1d_29/BiasAddBiasAdd!conv1d_29/conv1d/Squeeze:output:0(conv1d_29/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_29/BiasAddz
conv1d_29/ReluReluconv1d_29/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_29/Relu?
conv1d_30/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_30/Pad/paddings?
conv1d_30/PadPadconv1d_29/Relu:activations:0conv1d_30/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_30/Pad?
conv1d_30/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_30/conv1d/ExpandDims/dim?
conv1d_30/conv1d/ExpandDims
ExpandDimsconv1d_30/Pad:output:0(conv1d_30/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_30/conv1d/ExpandDims?
,conv1d_30/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_30_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_30/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_30/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_30/conv1d/ExpandDims_1/dim?
conv1d_30/conv1d/ExpandDims_1
ExpandDims4conv1d_30/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_30/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_30/conv1d/ExpandDims_1?
conv1d_30/conv1d	MLCConv2D$conv1d_30/conv1d/ExpandDims:output:0&conv1d_30/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_30/conv1d?
conv1d_30/conv1d/SqueezeSqueezeconv1d_30/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_30/conv1d/Squeeze?
 conv1d_30/BiasAdd/ReadVariableOpReadVariableOp)conv1d_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_30/BiasAdd/ReadVariableOp?
conv1d_30/BiasAddBiasAdd!conv1d_30/conv1d/Squeeze:output:0(conv1d_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_30/BiasAddz
conv1d_30/ReluReluconv1d_30/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_30/Relu?
conv1d_31/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_31/Pad/paddings?
conv1d_31/PadPadconv1d_30/Relu:activations:0conv1d_31/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_31/Pad?
conv1d_31/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_31/conv1d/ExpandDims/dim?
conv1d_31/conv1d/ExpandDims
ExpandDimsconv1d_31/Pad:output:0(conv1d_31/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_31/conv1d/ExpandDims?
,conv1d_31/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_31_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_31/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_31/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_31/conv1d/ExpandDims_1/dim?
conv1d_31/conv1d/ExpandDims_1
ExpandDims4conv1d_31/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_31/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_31/conv1d/ExpandDims_1?
conv1d_31/conv1d	MLCConv2D$conv1d_31/conv1d/ExpandDims:output:0&conv1d_31/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_31/conv1d?
conv1d_31/conv1d/SqueezeSqueezeconv1d_31/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_31/conv1d/Squeeze?
 conv1d_31/BiasAdd/ReadVariableOpReadVariableOp)conv1d_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_31/BiasAdd/ReadVariableOp?
conv1d_31/BiasAddBiasAdd!conv1d_31/conv1d/Squeeze:output:0(conv1d_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_31/BiasAddz
conv1d_31/ReluReluconv1d_31/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_31/Relu?
conv1d_32/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_32/Pad/paddings?
conv1d_32/PadPadconv1d_31/Relu:activations:0conv1d_32/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_32/Pad?
conv1d_32/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_32/conv1d/ExpandDims/dim?
conv1d_32/conv1d/ExpandDims
ExpandDimsconv1d_32/Pad:output:0(conv1d_32/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_32/conv1d/ExpandDims?
,conv1d_32/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_32_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_32/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_32/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_32/conv1d/ExpandDims_1/dim?
conv1d_32/conv1d/ExpandDims_1
ExpandDims4conv1d_32/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_32/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_32/conv1d/ExpandDims_1?
conv1d_32/conv1d	MLCConv2D$conv1d_32/conv1d/ExpandDims:output:0&conv1d_32/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_32/conv1d?
conv1d_32/conv1d/SqueezeSqueezeconv1d_32/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_32/conv1d/Squeeze?
 conv1d_32/BiasAdd/ReadVariableOpReadVariableOp)conv1d_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_32/BiasAdd/ReadVariableOp?
conv1d_32/BiasAddBiasAdd!conv1d_32/conv1d/Squeeze:output:0(conv1d_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_32/BiasAddz
conv1d_32/ReluReluconv1d_32/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_32/Relu?
conv1d_33/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_33/Pad/paddings?
conv1d_33/PadPadconv1d_32/Relu:activations:0conv1d_33/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_33/Pad?
conv1d_33/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_33/conv1d/ExpandDims/dim?
conv1d_33/conv1d/ExpandDims
ExpandDimsconv1d_33/Pad:output:0(conv1d_33/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_33/conv1d/ExpandDims?
,conv1d_33/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_33_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_33/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_33/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_33/conv1d/ExpandDims_1/dim?
conv1d_33/conv1d/ExpandDims_1
ExpandDims4conv1d_33/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_33/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_33/conv1d/ExpandDims_1?
conv1d_33/conv1d	MLCConv2D$conv1d_33/conv1d/ExpandDims:output:0&conv1d_33/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_33/conv1d?
conv1d_33/conv1d/SqueezeSqueezeconv1d_33/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_33/conv1d/Squeeze?
 conv1d_33/BiasAdd/ReadVariableOpReadVariableOp)conv1d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_33/BiasAdd/ReadVariableOp?
conv1d_33/BiasAddBiasAdd!conv1d_33/conv1d/Squeeze:output:0(conv1d_33/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_33/BiasAddz
conv1d_33/ReluReluconv1d_33/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_33/Relu?
conv1d_34/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_34/Pad/paddings?
conv1d_34/PadPadconv1d_33/Relu:activations:0conv1d_34/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_34/Pad?
conv1d_34/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_34/conv1d/ExpandDims/dim?
conv1d_34/conv1d/ExpandDims
ExpandDimsconv1d_34/Pad:output:0(conv1d_34/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_34/conv1d/ExpandDims?
,conv1d_34/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_34_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_34/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_34/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_34/conv1d/ExpandDims_1/dim?
conv1d_34/conv1d/ExpandDims_1
ExpandDims4conv1d_34/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_34/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_34/conv1d/ExpandDims_1?
conv1d_34/conv1d	MLCConv2D$conv1d_34/conv1d/ExpandDims:output:0&conv1d_34/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_34/conv1d?
conv1d_34/conv1d/SqueezeSqueezeconv1d_34/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_34/conv1d/Squeeze?
 conv1d_34/BiasAdd/ReadVariableOpReadVariableOp)conv1d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_34/BiasAdd/ReadVariableOp?
conv1d_34/BiasAddBiasAdd!conv1d_34/conv1d/Squeeze:output:0(conv1d_34/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_34/BiasAddz
conv1d_34/ReluReluconv1d_34/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_34/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_4/Const?
flatten_4/ReshapeReshapeconv1d_34/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
 dense_8/MLCMatMul/ReadVariableOpReadVariableOp)dense_8_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_8/MLCMatMul/ReadVariableOp?
dense_8/MLCMatMul	MLCMatMulflatten_4/Reshape:output:0(dense_8/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MLCMatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MLCMatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAddp
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_8/Relu?
 dense_9/MLCMatMul/ReadVariableOpReadVariableOp)dense_9_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_9/MLCMatMul/ReadVariableOp?
dense_9/MLCMatMul	MLCMatMuldense_8/Relu:activations:0(dense_9/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MLCMatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MLCMatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAddy
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_9/Softmax?
IdentityIdentitydense_9/Softmax:softmax:0!^conv1d_28/BiasAdd/ReadVariableOp-^conv1d_28/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_29/BiasAdd/ReadVariableOp-^conv1d_29/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_30/BiasAdd/ReadVariableOp-^conv1d_30/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_31/BiasAdd/ReadVariableOp-^conv1d_31/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_32/BiasAdd/ReadVariableOp-^conv1d_32/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_33/BiasAdd/ReadVariableOp-^conv1d_33/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_34/BiasAdd/ReadVariableOp-^conv1d_34/conv1d/ExpandDims_1/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/MLCMatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2D
 conv1d_28/BiasAdd/ReadVariableOp conv1d_28/BiasAdd/ReadVariableOp2\
,conv1d_28/conv1d/ExpandDims_1/ReadVariableOp,conv1d_28/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_29/BiasAdd/ReadVariableOp conv1d_29/BiasAdd/ReadVariableOp2\
,conv1d_29/conv1d/ExpandDims_1/ReadVariableOp,conv1d_29/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_30/BiasAdd/ReadVariableOp conv1d_30/BiasAdd/ReadVariableOp2\
,conv1d_30/conv1d/ExpandDims_1/ReadVariableOp,conv1d_30/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_31/BiasAdd/ReadVariableOp conv1d_31/BiasAdd/ReadVariableOp2\
,conv1d_31/conv1d/ExpandDims_1/ReadVariableOp,conv1d_31/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_32/BiasAdd/ReadVariableOp conv1d_32/BiasAdd/ReadVariableOp2\
,conv1d_32/conv1d/ExpandDims_1/ReadVariableOp,conv1d_32/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_33/BiasAdd/ReadVariableOp conv1d_33/BiasAdd/ReadVariableOp2\
,conv1d_33/conv1d/ExpandDims_1/ReadVariableOp,conv1d_33/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_34/BiasAdd/ReadVariableOp conv1d_34/BiasAdd/ReadVariableOp2\
,conv1d_34/conv1d/ExpandDims_1/ReadVariableOp,conv1d_34/conv1d/ExpandDims_1/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/MLCMatMul/ReadVariableOp dense_8/MLCMatMul/ReadVariableOp2@
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
:?????????
"
_user_specified_name
inputs/1
?
?
(__inference_model_4_layer_call_fn_422689
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
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_4222762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
?
?
$__inference_signature_wrapper_422367
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
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_4217652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
E__inference_conv1d_28_layer_call_and_return_conditional_losses_422707

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
?3
?
C__inference_model_4_layer_call_and_return_conditional_losses_422128
conv
cat
conv1d_28_422081
conv1d_28_422083
conv1d_29_422086
conv1d_29_422088
conv1d_30_422091
conv1d_30_422093
conv1d_31_422096
conv1d_31_422098
conv1d_32_422101
conv1d_32_422103
conv1d_33_422106
conv1d_33_422108
conv1d_34_422111
conv1d_34_422113
dense_8_422117
dense_8_422119
dense_9_422122
dense_9_422124
identity??!conv1d_28/StatefulPartitionedCall?!conv1d_29/StatefulPartitionedCall?!conv1d_30/StatefulPartitionedCall?!conv1d_31/StatefulPartitionedCall?!conv1d_32/StatefulPartitionedCall?!conv1d_33/StatefulPartitionedCall?!conv1d_34/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
!conv1d_28/StatefulPartitionedCallStatefulPartitionedCallconvconv1d_28_422081conv1d_28_422083*
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
E__inference_conv1d_28_layer_call_and_return_conditional_losses_4217882#
!conv1d_28/StatefulPartitionedCall?
!conv1d_29/StatefulPartitionedCallStatefulPartitionedCall*conv1d_28/StatefulPartitionedCall:output:0conv1d_29_422086conv1d_29_422088*
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
E__inference_conv1d_29_layer_call_and_return_conditional_losses_4218222#
!conv1d_29/StatefulPartitionedCall?
!conv1d_30/StatefulPartitionedCallStatefulPartitionedCall*conv1d_29/StatefulPartitionedCall:output:0conv1d_30_422091conv1d_30_422093*
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
E__inference_conv1d_30_layer_call_and_return_conditional_losses_4218562#
!conv1d_30/StatefulPartitionedCall?
!conv1d_31/StatefulPartitionedCallStatefulPartitionedCall*conv1d_30/StatefulPartitionedCall:output:0conv1d_31_422096conv1d_31_422098*
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
E__inference_conv1d_31_layer_call_and_return_conditional_losses_4218902#
!conv1d_31/StatefulPartitionedCall?
!conv1d_32/StatefulPartitionedCallStatefulPartitionedCall*conv1d_31/StatefulPartitionedCall:output:0conv1d_32_422101conv1d_32_422103*
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
E__inference_conv1d_32_layer_call_and_return_conditional_losses_4219242#
!conv1d_32/StatefulPartitionedCall?
!conv1d_33/StatefulPartitionedCallStatefulPartitionedCall*conv1d_32/StatefulPartitionedCall:output:0conv1d_33_422106conv1d_33_422108*
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
E__inference_conv1d_33_layer_call_and_return_conditional_losses_4219582#
!conv1d_33/StatefulPartitionedCall?
!conv1d_34/StatefulPartitionedCallStatefulPartitionedCall*conv1d_33/StatefulPartitionedCall:output:0conv1d_34_422111conv1d_34_422113*
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
E__inference_conv1d_34_layer_call_and_return_conditional_losses_4219922#
!conv1d_34/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCall*conv1d_34/StatefulPartitionedCall:output:0*
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
E__inference_flatten_4_layer_call_and_return_conditional_losses_4220142
flatten_4/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_8_422117dense_8_422119*
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
GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_4220332!
dense_8/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_422122dense_9_422124*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_4220602!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0"^conv1d_28/StatefulPartitionedCall"^conv1d_29/StatefulPartitionedCall"^conv1d_30/StatefulPartitionedCall"^conv1d_31/StatefulPartitionedCall"^conv1d_32/StatefulPartitionedCall"^conv1d_33/StatefulPartitionedCall"^conv1d_34/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2F
!conv1d_28/StatefulPartitionedCall!conv1d_28/StatefulPartitionedCall2F
!conv1d_29/StatefulPartitionedCall!conv1d_29/StatefulPartitionedCall2F
!conv1d_30/StatefulPartitionedCall!conv1d_30/StatefulPartitionedCall2F
!conv1d_31/StatefulPartitionedCall!conv1d_31/StatefulPartitionedCall2F
!conv1d_32/StatefulPartitionedCall!conv1d_32/StatefulPartitionedCall2F
!conv1d_33/StatefulPartitionedCall!conv1d_33/StatefulPartitionedCall2F
!conv1d_34/StatefulPartitionedCall!conv1d_34/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
?
?
E__inference_conv1d_34_layer_call_and_return_conditional_losses_421992

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
*__inference_conv1d_32_layer_call_fn_422824

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
E__inference_conv1d_32_layer_call_and_return_conditional_losses_4219242
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
?
(__inference_model_4_layer_call_fn_422222
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
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_4221832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
?
?
E__inference_conv1d_30_layer_call_and_return_conditional_losses_422761

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
(__inference_model_4_layer_call_fn_422315
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
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_4222762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
?
?
E__inference_conv1d_33_layer_call_and_return_conditional_losses_421958

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
?
!__inference__wrapped_model_421765
conv
catA
=model_4_conv1d_28_conv1d_expanddims_1_readvariableop_resource5
1model_4_conv1d_28_biasadd_readvariableop_resourceA
=model_4_conv1d_29_conv1d_expanddims_1_readvariableop_resource5
1model_4_conv1d_29_biasadd_readvariableop_resourceA
=model_4_conv1d_30_conv1d_expanddims_1_readvariableop_resource5
1model_4_conv1d_30_biasadd_readvariableop_resourceA
=model_4_conv1d_31_conv1d_expanddims_1_readvariableop_resource5
1model_4_conv1d_31_biasadd_readvariableop_resourceA
=model_4_conv1d_32_conv1d_expanddims_1_readvariableop_resource5
1model_4_conv1d_32_biasadd_readvariableop_resourceA
=model_4_conv1d_33_conv1d_expanddims_1_readvariableop_resource5
1model_4_conv1d_33_biasadd_readvariableop_resourceA
=model_4_conv1d_34_conv1d_expanddims_1_readvariableop_resource5
1model_4_conv1d_34_biasadd_readvariableop_resource5
1model_4_dense_8_mlcmatmul_readvariableop_resource3
/model_4_dense_8_biasadd_readvariableop_resource5
1model_4_dense_9_mlcmatmul_readvariableop_resource3
/model_4_dense_9_biasadd_readvariableop_resource
identity??(model_4/conv1d_28/BiasAdd/ReadVariableOp?4model_4/conv1d_28/conv1d/ExpandDims_1/ReadVariableOp?(model_4/conv1d_29/BiasAdd/ReadVariableOp?4model_4/conv1d_29/conv1d/ExpandDims_1/ReadVariableOp?(model_4/conv1d_30/BiasAdd/ReadVariableOp?4model_4/conv1d_30/conv1d/ExpandDims_1/ReadVariableOp?(model_4/conv1d_31/BiasAdd/ReadVariableOp?4model_4/conv1d_31/conv1d/ExpandDims_1/ReadVariableOp?(model_4/conv1d_32/BiasAdd/ReadVariableOp?4model_4/conv1d_32/conv1d/ExpandDims_1/ReadVariableOp?(model_4/conv1d_33/BiasAdd/ReadVariableOp?4model_4/conv1d_33/conv1d/ExpandDims_1/ReadVariableOp?(model_4/conv1d_34/BiasAdd/ReadVariableOp?4model_4/conv1d_34/conv1d/ExpandDims_1/ReadVariableOp?&model_4/dense_8/BiasAdd/ReadVariableOp?(model_4/dense_8/MLCMatMul/ReadVariableOp?&model_4/dense_9/BiasAdd/ReadVariableOp?(model_4/dense_9/MLCMatMul/ReadVariableOp?
model_4/conv1d_28/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_4/conv1d_28/Pad/paddings?
model_4/conv1d_28/PadPadconv'model_4/conv1d_28/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
model_4/conv1d_28/Pad?
'model_4/conv1d_28/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_4/conv1d_28/conv1d/ExpandDims/dim?
#model_4/conv1d_28/conv1d/ExpandDims
ExpandDimsmodel_4/conv1d_28/Pad:output:00model_4/conv1d_28/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2%
#model_4/conv1d_28/conv1d/ExpandDims?
4model_4/conv1d_28/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_4_conv1d_28_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype026
4model_4/conv1d_28/conv1d/ExpandDims_1/ReadVariableOp?
)model_4/conv1d_28/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_4/conv1d_28/conv1d/ExpandDims_1/dim?
%model_4/conv1d_28/conv1d/ExpandDims_1
ExpandDims<model_4/conv1d_28/conv1d/ExpandDims_1/ReadVariableOp:value:02model_4/conv1d_28/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2'
%model_4/conv1d_28/conv1d/ExpandDims_1?
model_4/conv1d_28/conv1d	MLCConv2D,model_4/conv1d_28/conv1d/ExpandDims:output:0.model_4/conv1d_28/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_4/conv1d_28/conv1d?
 model_4/conv1d_28/conv1d/SqueezeSqueeze!model_4/conv1d_28/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_4/conv1d_28/conv1d/Squeeze?
(model_4/conv1d_28/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv1d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_4/conv1d_28/BiasAdd/ReadVariableOp?
model_4/conv1d_28/BiasAddBiasAdd)model_4/conv1d_28/conv1d/Squeeze:output:00model_4/conv1d_28/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_28/BiasAdd?
model_4/conv1d_28/ReluRelu"model_4/conv1d_28/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_28/Relu?
model_4/conv1d_29/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_4/conv1d_29/Pad/paddings?
model_4/conv1d_29/PadPad$model_4/conv1d_28/Relu:activations:0'model_4/conv1d_29/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_29/Pad?
'model_4/conv1d_29/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_4/conv1d_29/conv1d/ExpandDims/dim?
#model_4/conv1d_29/conv1d/ExpandDims
ExpandDimsmodel_4/conv1d_29/Pad:output:00model_4/conv1d_29/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_4/conv1d_29/conv1d/ExpandDims?
4model_4/conv1d_29/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_4_conv1d_29_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_4/conv1d_29/conv1d/ExpandDims_1/ReadVariableOp?
)model_4/conv1d_29/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_4/conv1d_29/conv1d/ExpandDims_1/dim?
%model_4/conv1d_29/conv1d/ExpandDims_1
ExpandDims<model_4/conv1d_29/conv1d/ExpandDims_1/ReadVariableOp:value:02model_4/conv1d_29/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_4/conv1d_29/conv1d/ExpandDims_1?
model_4/conv1d_29/conv1d	MLCConv2D,model_4/conv1d_29/conv1d/ExpandDims:output:0.model_4/conv1d_29/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_4/conv1d_29/conv1d?
 model_4/conv1d_29/conv1d/SqueezeSqueeze!model_4/conv1d_29/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_4/conv1d_29/conv1d/Squeeze?
(model_4/conv1d_29/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv1d_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_4/conv1d_29/BiasAdd/ReadVariableOp?
model_4/conv1d_29/BiasAddBiasAdd)model_4/conv1d_29/conv1d/Squeeze:output:00model_4/conv1d_29/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_29/BiasAdd?
model_4/conv1d_29/ReluRelu"model_4/conv1d_29/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_29/Relu?
model_4/conv1d_30/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_4/conv1d_30/Pad/paddings?
model_4/conv1d_30/PadPad$model_4/conv1d_29/Relu:activations:0'model_4/conv1d_30/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_30/Pad?
'model_4/conv1d_30/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_4/conv1d_30/conv1d/ExpandDims/dim?
#model_4/conv1d_30/conv1d/ExpandDims
ExpandDimsmodel_4/conv1d_30/Pad:output:00model_4/conv1d_30/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_4/conv1d_30/conv1d/ExpandDims?
4model_4/conv1d_30/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_4_conv1d_30_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_4/conv1d_30/conv1d/ExpandDims_1/ReadVariableOp?
)model_4/conv1d_30/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_4/conv1d_30/conv1d/ExpandDims_1/dim?
%model_4/conv1d_30/conv1d/ExpandDims_1
ExpandDims<model_4/conv1d_30/conv1d/ExpandDims_1/ReadVariableOp:value:02model_4/conv1d_30/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_4/conv1d_30/conv1d/ExpandDims_1?
model_4/conv1d_30/conv1d	MLCConv2D,model_4/conv1d_30/conv1d/ExpandDims:output:0.model_4/conv1d_30/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_4/conv1d_30/conv1d?
 model_4/conv1d_30/conv1d/SqueezeSqueeze!model_4/conv1d_30/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_4/conv1d_30/conv1d/Squeeze?
(model_4/conv1d_30/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv1d_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_4/conv1d_30/BiasAdd/ReadVariableOp?
model_4/conv1d_30/BiasAddBiasAdd)model_4/conv1d_30/conv1d/Squeeze:output:00model_4/conv1d_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_30/BiasAdd?
model_4/conv1d_30/ReluRelu"model_4/conv1d_30/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_30/Relu?
model_4/conv1d_31/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_4/conv1d_31/Pad/paddings?
model_4/conv1d_31/PadPad$model_4/conv1d_30/Relu:activations:0'model_4/conv1d_31/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_31/Pad?
'model_4/conv1d_31/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_4/conv1d_31/conv1d/ExpandDims/dim?
#model_4/conv1d_31/conv1d/ExpandDims
ExpandDimsmodel_4/conv1d_31/Pad:output:00model_4/conv1d_31/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_4/conv1d_31/conv1d/ExpandDims?
4model_4/conv1d_31/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_4_conv1d_31_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_4/conv1d_31/conv1d/ExpandDims_1/ReadVariableOp?
)model_4/conv1d_31/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_4/conv1d_31/conv1d/ExpandDims_1/dim?
%model_4/conv1d_31/conv1d/ExpandDims_1
ExpandDims<model_4/conv1d_31/conv1d/ExpandDims_1/ReadVariableOp:value:02model_4/conv1d_31/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_4/conv1d_31/conv1d/ExpandDims_1?
model_4/conv1d_31/conv1d	MLCConv2D,model_4/conv1d_31/conv1d/ExpandDims:output:0.model_4/conv1d_31/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_4/conv1d_31/conv1d?
 model_4/conv1d_31/conv1d/SqueezeSqueeze!model_4/conv1d_31/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_4/conv1d_31/conv1d/Squeeze?
(model_4/conv1d_31/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv1d_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_4/conv1d_31/BiasAdd/ReadVariableOp?
model_4/conv1d_31/BiasAddBiasAdd)model_4/conv1d_31/conv1d/Squeeze:output:00model_4/conv1d_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_31/BiasAdd?
model_4/conv1d_31/ReluRelu"model_4/conv1d_31/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_31/Relu?
model_4/conv1d_32/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_4/conv1d_32/Pad/paddings?
model_4/conv1d_32/PadPad$model_4/conv1d_31/Relu:activations:0'model_4/conv1d_32/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_32/Pad?
'model_4/conv1d_32/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_4/conv1d_32/conv1d/ExpandDims/dim?
#model_4/conv1d_32/conv1d/ExpandDims
ExpandDimsmodel_4/conv1d_32/Pad:output:00model_4/conv1d_32/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_4/conv1d_32/conv1d/ExpandDims?
4model_4/conv1d_32/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_4_conv1d_32_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_4/conv1d_32/conv1d/ExpandDims_1/ReadVariableOp?
)model_4/conv1d_32/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_4/conv1d_32/conv1d/ExpandDims_1/dim?
%model_4/conv1d_32/conv1d/ExpandDims_1
ExpandDims<model_4/conv1d_32/conv1d/ExpandDims_1/ReadVariableOp:value:02model_4/conv1d_32/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_4/conv1d_32/conv1d/ExpandDims_1?
model_4/conv1d_32/conv1d	MLCConv2D,model_4/conv1d_32/conv1d/ExpandDims:output:0.model_4/conv1d_32/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_4/conv1d_32/conv1d?
 model_4/conv1d_32/conv1d/SqueezeSqueeze!model_4/conv1d_32/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_4/conv1d_32/conv1d/Squeeze?
(model_4/conv1d_32/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv1d_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_4/conv1d_32/BiasAdd/ReadVariableOp?
model_4/conv1d_32/BiasAddBiasAdd)model_4/conv1d_32/conv1d/Squeeze:output:00model_4/conv1d_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_32/BiasAdd?
model_4/conv1d_32/ReluRelu"model_4/conv1d_32/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_32/Relu?
model_4/conv1d_33/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_4/conv1d_33/Pad/paddings?
model_4/conv1d_33/PadPad$model_4/conv1d_32/Relu:activations:0'model_4/conv1d_33/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_33/Pad?
'model_4/conv1d_33/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_4/conv1d_33/conv1d/ExpandDims/dim?
#model_4/conv1d_33/conv1d/ExpandDims
ExpandDimsmodel_4/conv1d_33/Pad:output:00model_4/conv1d_33/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_4/conv1d_33/conv1d/ExpandDims?
4model_4/conv1d_33/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_4_conv1d_33_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_4/conv1d_33/conv1d/ExpandDims_1/ReadVariableOp?
)model_4/conv1d_33/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_4/conv1d_33/conv1d/ExpandDims_1/dim?
%model_4/conv1d_33/conv1d/ExpandDims_1
ExpandDims<model_4/conv1d_33/conv1d/ExpandDims_1/ReadVariableOp:value:02model_4/conv1d_33/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_4/conv1d_33/conv1d/ExpandDims_1?
model_4/conv1d_33/conv1d	MLCConv2D,model_4/conv1d_33/conv1d/ExpandDims:output:0.model_4/conv1d_33/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_4/conv1d_33/conv1d?
 model_4/conv1d_33/conv1d/SqueezeSqueeze!model_4/conv1d_33/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_4/conv1d_33/conv1d/Squeeze?
(model_4/conv1d_33/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv1d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_4/conv1d_33/BiasAdd/ReadVariableOp?
model_4/conv1d_33/BiasAddBiasAdd)model_4/conv1d_33/conv1d/Squeeze:output:00model_4/conv1d_33/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_33/BiasAdd?
model_4/conv1d_33/ReluRelu"model_4/conv1d_33/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_33/Relu?
model_4/conv1d_34/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_4/conv1d_34/Pad/paddings?
model_4/conv1d_34/PadPad$model_4/conv1d_33/Relu:activations:0'model_4/conv1d_34/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_34/Pad?
'model_4/conv1d_34/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_4/conv1d_34/conv1d/ExpandDims/dim?
#model_4/conv1d_34/conv1d/ExpandDims
ExpandDimsmodel_4/conv1d_34/Pad:output:00model_4/conv1d_34/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_4/conv1d_34/conv1d/ExpandDims?
4model_4/conv1d_34/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_4_conv1d_34_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_4/conv1d_34/conv1d/ExpandDims_1/ReadVariableOp?
)model_4/conv1d_34/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_4/conv1d_34/conv1d/ExpandDims_1/dim?
%model_4/conv1d_34/conv1d/ExpandDims_1
ExpandDims<model_4/conv1d_34/conv1d/ExpandDims_1/ReadVariableOp:value:02model_4/conv1d_34/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_4/conv1d_34/conv1d/ExpandDims_1?
model_4/conv1d_34/conv1d	MLCConv2D,model_4/conv1d_34/conv1d/ExpandDims:output:0.model_4/conv1d_34/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_4/conv1d_34/conv1d?
 model_4/conv1d_34/conv1d/SqueezeSqueeze!model_4/conv1d_34/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_4/conv1d_34/conv1d/Squeeze?
(model_4/conv1d_34/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv1d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_4/conv1d_34/BiasAdd/ReadVariableOp?
model_4/conv1d_34/BiasAddBiasAdd)model_4/conv1d_34/conv1d/Squeeze:output:00model_4/conv1d_34/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_34/BiasAdd?
model_4/conv1d_34/ReluRelu"model_4/conv1d_34/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_4/conv1d_34/Relu?
model_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model_4/flatten_4/Const?
model_4/flatten_4/ReshapeReshape$model_4/conv1d_34/Relu:activations:0 model_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
model_4/flatten_4/Reshape?
(model_4/dense_8/MLCMatMul/ReadVariableOpReadVariableOp1model_4_dense_8_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(model_4/dense_8/MLCMatMul/ReadVariableOp?
model_4/dense_8/MLCMatMul	MLCMatMul"model_4/flatten_4/Reshape:output:00model_4/dense_8/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/dense_8/MLCMatMul?
&model_4/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_4_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_4/dense_8/BiasAdd/ReadVariableOp?
model_4/dense_8/BiasAddBiasAdd#model_4/dense_8/MLCMatMul:product:0.model_4/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/dense_8/BiasAdd?
model_4/dense_8/ReluRelu model_4/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_4/dense_8/Relu?
(model_4/dense_9/MLCMatMul/ReadVariableOpReadVariableOp1model_4_dense_9_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(model_4/dense_9/MLCMatMul/ReadVariableOp?
model_4/dense_9/MLCMatMul	MLCMatMul"model_4/dense_8/Relu:activations:00model_4/dense_9/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/dense_9/MLCMatMul?
&model_4/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_4_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_4/dense_9/BiasAdd/ReadVariableOp?
model_4/dense_9/BiasAddBiasAdd#model_4/dense_9/MLCMatMul:product:0.model_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/dense_9/BiasAdd?
model_4/dense_9/SoftmaxSoftmax model_4/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_4/dense_9/Softmax?
IdentityIdentity!model_4/dense_9/Softmax:softmax:0)^model_4/conv1d_28/BiasAdd/ReadVariableOp5^model_4/conv1d_28/conv1d/ExpandDims_1/ReadVariableOp)^model_4/conv1d_29/BiasAdd/ReadVariableOp5^model_4/conv1d_29/conv1d/ExpandDims_1/ReadVariableOp)^model_4/conv1d_30/BiasAdd/ReadVariableOp5^model_4/conv1d_30/conv1d/ExpandDims_1/ReadVariableOp)^model_4/conv1d_31/BiasAdd/ReadVariableOp5^model_4/conv1d_31/conv1d/ExpandDims_1/ReadVariableOp)^model_4/conv1d_32/BiasAdd/ReadVariableOp5^model_4/conv1d_32/conv1d/ExpandDims_1/ReadVariableOp)^model_4/conv1d_33/BiasAdd/ReadVariableOp5^model_4/conv1d_33/conv1d/ExpandDims_1/ReadVariableOp)^model_4/conv1d_34/BiasAdd/ReadVariableOp5^model_4/conv1d_34/conv1d/ExpandDims_1/ReadVariableOp'^model_4/dense_8/BiasAdd/ReadVariableOp)^model_4/dense_8/MLCMatMul/ReadVariableOp'^model_4/dense_9/BiasAdd/ReadVariableOp)^model_4/dense_9/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????::::::::::::::::::2T
(model_4/conv1d_28/BiasAdd/ReadVariableOp(model_4/conv1d_28/BiasAdd/ReadVariableOp2l
4model_4/conv1d_28/conv1d/ExpandDims_1/ReadVariableOp4model_4/conv1d_28/conv1d/ExpandDims_1/ReadVariableOp2T
(model_4/conv1d_29/BiasAdd/ReadVariableOp(model_4/conv1d_29/BiasAdd/ReadVariableOp2l
4model_4/conv1d_29/conv1d/ExpandDims_1/ReadVariableOp4model_4/conv1d_29/conv1d/ExpandDims_1/ReadVariableOp2T
(model_4/conv1d_30/BiasAdd/ReadVariableOp(model_4/conv1d_30/BiasAdd/ReadVariableOp2l
4model_4/conv1d_30/conv1d/ExpandDims_1/ReadVariableOp4model_4/conv1d_30/conv1d/ExpandDims_1/ReadVariableOp2T
(model_4/conv1d_31/BiasAdd/ReadVariableOp(model_4/conv1d_31/BiasAdd/ReadVariableOp2l
4model_4/conv1d_31/conv1d/ExpandDims_1/ReadVariableOp4model_4/conv1d_31/conv1d/ExpandDims_1/ReadVariableOp2T
(model_4/conv1d_32/BiasAdd/ReadVariableOp(model_4/conv1d_32/BiasAdd/ReadVariableOp2l
4model_4/conv1d_32/conv1d/ExpandDims_1/ReadVariableOp4model_4/conv1d_32/conv1d/ExpandDims_1/ReadVariableOp2T
(model_4/conv1d_33/BiasAdd/ReadVariableOp(model_4/conv1d_33/BiasAdd/ReadVariableOp2l
4model_4/conv1d_33/conv1d/ExpandDims_1/ReadVariableOp4model_4/conv1d_33/conv1d/ExpandDims_1/ReadVariableOp2T
(model_4/conv1d_34/BiasAdd/ReadVariableOp(model_4/conv1d_34/BiasAdd/ReadVariableOp2l
4model_4/conv1d_34/conv1d/ExpandDims_1/ReadVariableOp4model_4/conv1d_34/conv1d/ExpandDims_1/ReadVariableOp2P
&model_4/dense_8/BiasAdd/ReadVariableOp&model_4/dense_8/BiasAdd/ReadVariableOp2T
(model_4/dense_8/MLCMatMul/ReadVariableOp(model_4/dense_8/MLCMatMul/ReadVariableOp2P
&model_4/dense_9/BiasAdd/ReadVariableOp&model_4/dense_9/BiasAdd/ReadVariableOp2T
(model_4/dense_9/MLCMatMul/ReadVariableOp(model_4/dense_9/MLCMatMul/ReadVariableOp:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
?
}
(__inference_dense_9_layer_call_fn_422929

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_4220602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
*__inference_flatten_4_layer_call_fn_422889

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
E__inference_flatten_4_layer_call_and_return_conditional_losses_4220142
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
serving_default_conv:0?????????;
dense_90
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?v
_tf_keras_network?v{"class_name": "Functional", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv"}, "name": "conv", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_28", "inbound_nodes": [[["conv", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_29", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_29", "inbound_nodes": [[["conv1d_28", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_30", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_30", "inbound_nodes": [[["conv1d_29", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_31", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_31", "inbound_nodes": [[["conv1d_30", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_32", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_32", "inbound_nodes": [[["conv1d_31", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_33", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_33", "inbound_nodes": [[["conv1d_32", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_34", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_34", "inbound_nodes": [[["conv1d_33", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["conv1d_34", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 19, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["flatten_4", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cat"}, "name": "cat", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 8, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}], "input_layers": [["conv", 0, 0], ["cat", 0, 0]], "output_layers": [["dense_9", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 26, 1]}, {"class_name": "TensorShape", "items": [null, 3]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv"}, "name": "conv", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_28", "inbound_nodes": [[["conv", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_29", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_29", "inbound_nodes": [[["conv1d_28", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_30", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_30", "inbound_nodes": [[["conv1d_29", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_31", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_31", "inbound_nodes": [[["conv1d_30", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_32", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_32", "inbound_nodes": [[["conv1d_31", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_33", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_33", "inbound_nodes": [[["conv1d_32", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_34", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_34", "inbound_nodes": [[["conv1d_33", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["conv1d_34", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 19, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["flatten_4", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cat"}, "name": "cat", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 8, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}], "input_layers": [["conv", 0, 0], ["cat", 0, 0]], "output_layers": [["dense_9", 0, 0]]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv"}}
?


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv1D", "name": "conv1d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 1]}}
?	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_29", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

kernel
 bias
!	variables
"regularization_losses
#trainable_variables
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_30", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

%kernel
&bias
'	variables
(regularization_losses
)trainable_variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_31", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

+kernel
,bias
-	variables
.regularization_losses
/trainable_variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_32", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

1kernel
2bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_33", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

7kernel
8bias
9	variables
:regularization_losses
;trainable_variables
<	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_34", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?
=	variables
>regularization_losses
?trainable_variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Akernel
Bbias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 19, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1664}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1664]}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cat", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cat"}}
?

Gkernel
Hbias
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 8, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 19}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19]}}
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
Rmetrics
Snon_trainable_variables

Tlayers
	variables
Ulayer_regularization_losses
regularization_losses
trainable_variables
Vlayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
&:$@2conv1d_28/kernel
:@2conv1d_28/bias
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
Wmetrics
Xnon_trainable_variables

Ylayers
Zlayer_regularization_losses
	variables
regularization_losses
trainable_variables
[layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_29/kernel
:@2conv1d_29/bias
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
\metrics
]non_trainable_variables

^layers
_layer_regularization_losses
	variables
regularization_losses
trainable_variables
`layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_30/kernel
:@2conv1d_30/bias
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
ametrics
bnon_trainable_variables

clayers
dlayer_regularization_losses
!	variables
"regularization_losses
#trainable_variables
elayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_31/kernel
:@2conv1d_31/bias
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
fmetrics
gnon_trainable_variables

hlayers
ilayer_regularization_losses
'	variables
(regularization_losses
)trainable_variables
jlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_32/kernel
:@2conv1d_32/bias
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
kmetrics
lnon_trainable_variables

mlayers
nlayer_regularization_losses
-	variables
.regularization_losses
/trainable_variables
olayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_33/kernel
:@2conv1d_33/bias
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
pmetrics
qnon_trainable_variables

rlayers
slayer_regularization_losses
3	variables
4regularization_losses
5trainable_variables
tlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_34/kernel
:@2conv1d_34/bias
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
umetrics
vnon_trainable_variables

wlayers
xlayer_regularization_losses
9	variables
:regularization_losses
;trainable_variables
ylayer_metrics
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
zmetrics
{non_trainable_variables

|layers
}layer_regularization_losses
=	variables
>regularization_losses
?trainable_variables
~layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_8/kernel
:2dense_8/bias
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
metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
C	variables
Dregularization_losses
Etrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_9/kernel
:2dense_9/bias
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
?metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
I	variables
Jregularization_losses
Ktrainable_variables
?layer_metrics
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
0
?0
?1"
trackable_list_wrapper
 "
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
+:)@2Adam/conv1d_28/kernel/m
!:@2Adam/conv1d_28/bias/m
+:)@@2Adam/conv1d_29/kernel/m
!:@2Adam/conv1d_29/bias/m
+:)@@2Adam/conv1d_30/kernel/m
!:@2Adam/conv1d_30/bias/m
+:)@@2Adam/conv1d_31/kernel/m
!:@2Adam/conv1d_31/bias/m
+:)@@2Adam/conv1d_32/kernel/m
!:@2Adam/conv1d_32/bias/m
+:)@@2Adam/conv1d_33/kernel/m
!:@2Adam/conv1d_33/bias/m
+:)@@2Adam/conv1d_34/kernel/m
!:@2Adam/conv1d_34/bias/m
&:$	?2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
%:#2Adam/dense_9/kernel/m
:2Adam/dense_9/bias/m
+:)@2Adam/conv1d_28/kernel/v
!:@2Adam/conv1d_28/bias/v
+:)@@2Adam/conv1d_29/kernel/v
!:@2Adam/conv1d_29/bias/v
+:)@@2Adam/conv1d_30/kernel/v
!:@2Adam/conv1d_30/bias/v
+:)@@2Adam/conv1d_31/kernel/v
!:@2Adam/conv1d_31/bias/v
+:)@@2Adam/conv1d_32/kernel/v
!:@2Adam/conv1d_32/bias/v
+:)@@2Adam/conv1d_33/kernel/v
!:@2Adam/conv1d_33/bias/v
+:)@@2Adam/conv1d_34/kernel/v
!:@2Adam/conv1d_34/bias/v
&:$	?2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
%:#2Adam/dense_9/kernel/v
:2Adam/dense_9/bias/v
?2?
C__inference_model_4_layer_call_and_return_conditional_losses_422128
C__inference_model_4_layer_call_and_return_conditional_losses_422605
C__inference_model_4_layer_call_and_return_conditional_losses_422486
C__inference_model_4_layer_call_and_return_conditional_losses_422077?
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
!__inference__wrapped_model_421765?
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
?2?
(__inference_model_4_layer_call_fn_422689
(__inference_model_4_layer_call_fn_422647
(__inference_model_4_layer_call_fn_422315
(__inference_model_4_layer_call_fn_422222?
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
E__inference_conv1d_28_layer_call_and_return_conditional_losses_422707?
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
*__inference_conv1d_28_layer_call_fn_422716?
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
E__inference_conv1d_29_layer_call_and_return_conditional_losses_422734?
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
*__inference_conv1d_29_layer_call_fn_422743?
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
E__inference_conv1d_30_layer_call_and_return_conditional_losses_422761?
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
*__inference_conv1d_30_layer_call_fn_422770?
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
E__inference_conv1d_31_layer_call_and_return_conditional_losses_422788?
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
*__inference_conv1d_31_layer_call_fn_422797?
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
E__inference_conv1d_32_layer_call_and_return_conditional_losses_422815?
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
*__inference_conv1d_32_layer_call_fn_422824?
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
E__inference_conv1d_33_layer_call_and_return_conditional_losses_422842?
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
*__inference_conv1d_33_layer_call_fn_422851?
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
E__inference_conv1d_34_layer_call_and_return_conditional_losses_422869?
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
*__inference_conv1d_34_layer_call_fn_422878?
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
E__inference_flatten_4_layer_call_and_return_conditional_losses_422884?
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
*__inference_flatten_4_layer_call_fn_422889?
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
C__inference_dense_8_layer_call_and_return_conditional_losses_422900?
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
(__inference_dense_8_layer_call_fn_422909?
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
C__inference_dense_9_layer_call_and_return_conditional_losses_422920?
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
(__inference_dense_9_layer_call_fn_422929?
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
$__inference_signature_wrapper_422367catconv"?
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
!__inference__wrapped_model_421765? %&+,1278ABGHU?R
K?H
F?C
"?
conv?????????
?
cat?????????
? "1?.
,
dense_9!?
dense_9??????????
E__inference_conv1d_28_layer_call_and_return_conditional_losses_422707d3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????@
? ?
*__inference_conv1d_28_layer_call_fn_422716W3?0
)?&
$?!
inputs?????????
? "??????????@?
E__inference_conv1d_29_layer_call_and_return_conditional_losses_422734d3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_29_layer_call_fn_422743W3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_30_layer_call_and_return_conditional_losses_422761d 3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_30_layer_call_fn_422770W 3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_31_layer_call_and_return_conditional_losses_422788d%&3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_31_layer_call_fn_422797W%&3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_32_layer_call_and_return_conditional_losses_422815d+,3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_32_layer_call_fn_422824W+,3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_33_layer_call_and_return_conditional_losses_422842d123?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_33_layer_call_fn_422851W123?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_34_layer_call_and_return_conditional_losses_422869d783?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_34_layer_call_fn_422878W783?0
)?&
$?!
inputs?????????@
? "??????????@?
C__inference_dense_8_layer_call_and_return_conditional_losses_422900]AB0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_8_layer_call_fn_422909PAB0?-
&?#
!?
inputs??????????
? "???????????
C__inference_dense_9_layer_call_and_return_conditional_losses_422920\GH/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_9_layer_call_fn_422929OGH/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_4_layer_call_and_return_conditional_losses_422884]3?0
)?&
$?!
inputs?????????@
? "&?#
?
0??????????
? ~
*__inference_flatten_4_layer_call_fn_422889P3?0
)?&
$?!
inputs?????????@
? "????????????
C__inference_model_4_layer_call_and_return_conditional_losses_422077? %&+,1278ABGH]?Z
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
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_422128? %&+,1278ABGH]?Z
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
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_422486? %&+,1278ABGHf?c
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
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_422605? %&+,1278ABGHf?c
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
0?????????
? ?
(__inference_model_4_layer_call_fn_422222? %&+,1278ABGH]?Z
S?P
F?C
"?
conv?????????
?
cat?????????
p

 
? "???????????
(__inference_model_4_layer_call_fn_422315? %&+,1278ABGH]?Z
S?P
F?C
"?
conv?????????
?
cat?????????
p 

 
? "???????????
(__inference_model_4_layer_call_fn_422647? %&+,1278ABGHf?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
(__inference_model_4_layer_call_fn_422689? %&+,1278ABGHf?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
$__inference_signature_wrapper_422367? %&+,1278ABGH_?\
? 
U?R
$
cat?
cat?????????
*
conv"?
conv?????????"1?.
,
dense_9!?
dense_9?????????