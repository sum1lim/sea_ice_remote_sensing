??
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
conv1d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv1d_42/kernel
y
$conv1d_42/kernel/Read/ReadVariableOpReadVariableOpconv1d_42/kernel*"
_output_shapes
:@*
dtype0
t
conv1d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_42/bias
m
"conv1d_42/bias/Read/ReadVariableOpReadVariableOpconv1d_42/bias*
_output_shapes
:@*
dtype0
?
conv1d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_43/kernel
y
$conv1d_43/kernel/Read/ReadVariableOpReadVariableOpconv1d_43/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_43/bias
m
"conv1d_43/bias/Read/ReadVariableOpReadVariableOpconv1d_43/bias*
_output_shapes
:@*
dtype0
?
conv1d_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_44/kernel
y
$conv1d_44/kernel/Read/ReadVariableOpReadVariableOpconv1d_44/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_44/bias
m
"conv1d_44/bias/Read/ReadVariableOpReadVariableOpconv1d_44/bias*
_output_shapes
:@*
dtype0
?
conv1d_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_45/kernel
y
$conv1d_45/kernel/Read/ReadVariableOpReadVariableOpconv1d_45/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_45/bias
m
"conv1d_45/bias/Read/ReadVariableOpReadVariableOpconv1d_45/bias*
_output_shapes
:@*
dtype0
?
conv1d_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_46/kernel
y
$conv1d_46/kernel/Read/ReadVariableOpReadVariableOpconv1d_46/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_46/bias
m
"conv1d_46/bias/Read/ReadVariableOpReadVariableOpconv1d_46/bias*
_output_shapes
:@*
dtype0
?
conv1d_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_47/kernel
y
$conv1d_47/kernel/Read/ReadVariableOpReadVariableOpconv1d_47/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_47/bias
m
"conv1d_47/bias/Read/ReadVariableOpReadVariableOpconv1d_47/bias*
_output_shapes
:@*
dtype0
?
conv1d_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv1d_48/kernel
y
$conv1d_48/kernel/Read/ReadVariableOpReadVariableOpconv1d_48/kernel*"
_output_shapes
:@@*
dtype0
t
conv1d_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_48/bias
m
"conv1d_48/bias/Read/ReadVariableOpReadVariableOpconv1d_48/bias*
_output_shapes
:@*
dtype0
z
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

:*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:*
dtype0
{
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_19/kernel
t
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes
:	?*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:*
dtype0
z
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
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
Adam/conv1d_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv1d_42/kernel/m
?
+Adam/conv1d_42/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_42/kernel/m*"
_output_shapes
:@*
dtype0
?
Adam/conv1d_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_42/bias/m
{
)Adam/conv1d_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_42/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_43/kernel/m
?
+Adam/conv1d_43/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_43/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_43/bias/m
{
)Adam/conv1d_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_43/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_44/kernel/m
?
+Adam/conv1d_44/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_44/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_44/bias/m
{
)Adam/conv1d_44/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_44/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_45/kernel/m
?
+Adam/conv1d_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_45/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_45/bias/m
{
)Adam/conv1d_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_45/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_46/kernel/m
?
+Adam/conv1d_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_46/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_46/bias/m
{
)Adam/conv1d_46/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_46/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_47/kernel/m
?
+Adam/conv1d_47/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_47/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_47/bias/m
{
)Adam/conv1d_47/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_47/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_48/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_48/kernel/m
?
+Adam/conv1d_48/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_48/kernel/m*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_48/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_48/bias/m
{
)Adam/conv1d_48/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_48/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_18/kernel/m
?
*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_18/bias/m
y
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_19/kernel/m
?
*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/m
y
(Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_20/kernel/m
?
*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_20/bias/m
y
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1d_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv1d_42/kernel/v
?
+Adam/conv1d_42/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_42/kernel/v*"
_output_shapes
:@*
dtype0
?
Adam/conv1d_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_42/bias/v
{
)Adam/conv1d_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_42/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_43/kernel/v
?
+Adam/conv1d_43/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_43/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_43/bias/v
{
)Adam/conv1d_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_43/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_44/kernel/v
?
+Adam/conv1d_44/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_44/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_44/bias/v
{
)Adam/conv1d_44/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_44/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_45/kernel/v
?
+Adam/conv1d_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_45/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_45/bias/v
{
)Adam/conv1d_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_45/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_46/kernel/v
?
+Adam/conv1d_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_46/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_46/bias/v
{
)Adam/conv1d_46/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_46/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_47/kernel/v
?
+Adam/conv1d_47/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_47/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_47/bias/v
{
)Adam/conv1d_47/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_47/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_48/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv1d_48/kernel/v
?
+Adam/conv1d_48/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_48/kernel/v*"
_output_shapes
:@@*
dtype0
?
Adam/conv1d_48/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_48/bias/v
{
)Adam/conv1d_48/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_48/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_18/kernel/v
?
*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_18/bias/v
y
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_19/kernel/v
?
*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/v
y
(Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_20/kernel/v
?
*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_20/bias/v
y
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes
:*
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
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
 
h

9kernel
:bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
h

?kernel
@bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
R
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
R
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
h

Mkernel
Nbias
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
h

Skernel
Tbias
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
?
Yiter

Zbeta_1

[beta_2
	\decay
]learning_ratem?m?m?m?!m?"m?'m?(m?-m?.m?3m?4m?9m?:m??m?@m?Mm?Nm?Sm?Tm?v?v?v?v?!v?"v?'v?(v?-v?.v?3v?4v?9v?:v??v?@v?Mv?Nv?Sv?Tv?
 
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
?
regularization_losses
^metrics
_layer_metrics

`layers
anon_trainable_variables
blayer_regularization_losses
trainable_variables
	variables
 
\Z
VARIABLE_VALUEconv1d_42/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_42/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
cmetrics
dlayer_metrics

elayers
fnon_trainable_variables
glayer_regularization_losses
trainable_variables
	variables
\Z
VARIABLE_VALUEconv1d_43/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_43/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
hmetrics
ilayer_metrics

jlayers
knon_trainable_variables
llayer_regularization_losses
trainable_variables
	variables
\Z
VARIABLE_VALUEconv1d_44/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_44/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
?
#regularization_losses
mmetrics
nlayer_metrics

olayers
pnon_trainable_variables
qlayer_regularization_losses
$trainable_variables
%	variables
\Z
VARIABLE_VALUEconv1d_45/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_45/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
?
)regularization_losses
rmetrics
slayer_metrics

tlayers
unon_trainable_variables
vlayer_regularization_losses
*trainable_variables
+	variables
\Z
VARIABLE_VALUEconv1d_46/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_46/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
?
/regularization_losses
wmetrics
xlayer_metrics

ylayers
znon_trainable_variables
{layer_regularization_losses
0trainable_variables
1	variables
\Z
VARIABLE_VALUEconv1d_47/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_47/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
?
5regularization_losses
|metrics
}layer_metrics

~layers
non_trainable_variables
 ?layer_regularization_losses
6trainable_variables
7	variables
\Z
VARIABLE_VALUEconv1d_48/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_48/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

90
:1
?
;regularization_losses
?metrics
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
<trainable_variables
=	variables
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1

?0
@1
?
Aregularization_losses
?metrics
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
Btrainable_variables
C	variables
 
 
 
?
Eregularization_losses
?metrics
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
Ftrainable_variables
G	variables
 
 
 
?
Iregularization_losses
?metrics
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
Jtrainable_variables
K	variables
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

M0
N1

M0
N1
?
Oregularization_losses
?metrics
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
Ptrainable_variables
Q	variables
[Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_20/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

S0
T1

S0
T1
?
Uregularization_losses
?metrics
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
Vtrainable_variables
W	variables
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
VARIABLE_VALUEAdam/conv1d_42/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_42/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_43/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_43/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_44/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_44/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_45/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_45/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_46/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_46/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_47/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_47/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_48/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_48/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_42/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_42/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_43/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_43/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_44/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_44/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_45/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_45/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_46/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_46/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_47/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_47/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_48/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_48/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_catserving_default_convconv1d_42/kernelconv1d_42/biasconv1d_43/kernelconv1d_43/biasconv1d_44/kernelconv1d_44/biasconv1d_45/kernelconv1d_45/biasconv1d_46/kernelconv1d_46/biasconv1d_47/kernelconv1d_47/biasconv1d_48/kernelconv1d_48/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/bias*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_867268
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv1d_42/kernel/Read/ReadVariableOp"conv1d_42/bias/Read/ReadVariableOp$conv1d_43/kernel/Read/ReadVariableOp"conv1d_43/bias/Read/ReadVariableOp$conv1d_44/kernel/Read/ReadVariableOp"conv1d_44/bias/Read/ReadVariableOp$conv1d_45/kernel/Read/ReadVariableOp"conv1d_45/bias/Read/ReadVariableOp$conv1d_46/kernel/Read/ReadVariableOp"conv1d_46/bias/Read/ReadVariableOp$conv1d_47/kernel/Read/ReadVariableOp"conv1d_47/bias/Read/ReadVariableOp$conv1d_48/kernel/Read/ReadVariableOp"conv1d_48/bias/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv1d_42/kernel/m/Read/ReadVariableOp)Adam/conv1d_42/bias/m/Read/ReadVariableOp+Adam/conv1d_43/kernel/m/Read/ReadVariableOp)Adam/conv1d_43/bias/m/Read/ReadVariableOp+Adam/conv1d_44/kernel/m/Read/ReadVariableOp)Adam/conv1d_44/bias/m/Read/ReadVariableOp+Adam/conv1d_45/kernel/m/Read/ReadVariableOp)Adam/conv1d_45/bias/m/Read/ReadVariableOp+Adam/conv1d_46/kernel/m/Read/ReadVariableOp)Adam/conv1d_46/bias/m/Read/ReadVariableOp+Adam/conv1d_47/kernel/m/Read/ReadVariableOp)Adam/conv1d_47/bias/m/Read/ReadVariableOp+Adam/conv1d_48/kernel/m/Read/ReadVariableOp)Adam/conv1d_48/bias/m/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp(Adam/dense_19/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp+Adam/conv1d_42/kernel/v/Read/ReadVariableOp)Adam/conv1d_42/bias/v/Read/ReadVariableOp+Adam/conv1d_43/kernel/v/Read/ReadVariableOp)Adam/conv1d_43/bias/v/Read/ReadVariableOp+Adam/conv1d_44/kernel/v/Read/ReadVariableOp)Adam/conv1d_44/bias/v/Read/ReadVariableOp+Adam/conv1d_45/kernel/v/Read/ReadVariableOp)Adam/conv1d_45/bias/v/Read/ReadVariableOp+Adam/conv1d_46/kernel/v/Read/ReadVariableOp)Adam/conv1d_46/bias/v/Read/ReadVariableOp+Adam/conv1d_47/kernel/v/Read/ReadVariableOp)Adam/conv1d_47/bias/v/Read/ReadVariableOp+Adam/conv1d_48/kernel/v/Read/ReadVariableOp)Adam/conv1d_48/bias/v/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp(Adam/dense_19/bias/v/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOpConst*R
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
__inference__traced_save_868120
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_42/kernelconv1d_42/biasconv1d_43/kernelconv1d_43/biasconv1d_44/kernelconv1d_44/biasconv1d_45/kernelconv1d_45/biasconv1d_46/kernelconv1d_46/biasconv1d_47/kernelconv1d_47/biasconv1d_48/kernelconv1d_48/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv1d_42/kernel/mAdam/conv1d_42/bias/mAdam/conv1d_43/kernel/mAdam/conv1d_43/bias/mAdam/conv1d_44/kernel/mAdam/conv1d_44/bias/mAdam/conv1d_45/kernel/mAdam/conv1d_45/bias/mAdam/conv1d_46/kernel/mAdam/conv1d_46/bias/mAdam/conv1d_47/kernel/mAdam/conv1d_47/bias/mAdam/conv1d_48/kernel/mAdam/conv1d_48/bias/mAdam/dense_18/kernel/mAdam/dense_18/bias/mAdam/dense_19/kernel/mAdam/dense_19/bias/mAdam/dense_20/kernel/mAdam/dense_20/bias/mAdam/conv1d_42/kernel/vAdam/conv1d_42/bias/vAdam/conv1d_43/kernel/vAdam/conv1d_43/bias/vAdam/conv1d_44/kernel/vAdam/conv1d_44/bias/vAdam/conv1d_45/kernel/vAdam/conv1d_45/bias/vAdam/conv1d_46/kernel/vAdam/conv1d_46/bias/vAdam/conv1d_47/kernel/vAdam/conv1d_47/bias/vAdam/conv1d_48/kernel/vAdam/conv1d_48/bias/vAdam/dense_18/kernel/vAdam/dense_18/bias/vAdam/dense_19/kernel/vAdam/dense_19/bias/vAdam/dense_20/kernel/vAdam/dense_20/bias/v*Q
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
"__inference__traced_restore_868337ʹ
?

*__inference_conv1d_43_layer_call_fn_867670

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
E__inference_conv1d_43_layer_call_and_return_conditional_losses_8666502
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
E__inference_conv1d_46_layer_call_and_return_conditional_losses_866752

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
*__inference_conv1d_46_layer_call_fn_867751

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
E__inference_conv1d_46_layer_call_and_return_conditional_losses_8667522
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
D__inference_dense_19_layer_call_and_return_conditional_losses_867860

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?
~
)__inference_dense_18_layer_call_fn_867825

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
GPU 2J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_8668472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_dense_18_layer_call_and_return_conditional_losses_866847

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
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
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?
?
E__inference_conv1d_44_layer_call_and_return_conditional_losses_867688

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
E__inference_conv1d_46_layer_call_and_return_conditional_losses_867742

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
E__inference_conv1d_42_layer_call_and_return_conditional_losses_867634

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
E__inference_conv1d_43_layer_call_and_return_conditional_losses_866650

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
(__inference_model_6_layer_call_fn_867212
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
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_8671692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
*__inference_conv1d_47_layer_call_fn_867778

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
E__inference_conv1d_47_layer_call_and_return_conditional_losses_8667862
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
D__inference_dense_20_layer_call_and_return_conditional_losses_867880

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?;
?
C__inference_model_6_layer_call_and_return_conditional_losses_867066

inputs
inputs_1
conv1d_42_867013
conv1d_42_867015
conv1d_43_867018
conv1d_43_867020
conv1d_44_867023
conv1d_44_867025
conv1d_45_867028
conv1d_45_867030
conv1d_46_867033
conv1d_46_867035
conv1d_47_867038
conv1d_47_867040
conv1d_48_867043
conv1d_48_867045
dense_18_867048
dense_18_867050
dense_19_867055
dense_19_867057
dense_20_867060
dense_20_867062
identity??!conv1d_42/StatefulPartitionedCall?!conv1d_43/StatefulPartitionedCall?!conv1d_44/StatefulPartitionedCall?!conv1d_45/StatefulPartitionedCall?!conv1d_46/StatefulPartitionedCall?!conv1d_47/StatefulPartitionedCall?!conv1d_48/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall?
!conv1d_42/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_42_867013conv1d_42_867015*
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
E__inference_conv1d_42_layer_call_and_return_conditional_losses_8666162#
!conv1d_42/StatefulPartitionedCall?
!conv1d_43/StatefulPartitionedCallStatefulPartitionedCall*conv1d_42/StatefulPartitionedCall:output:0conv1d_43_867018conv1d_43_867020*
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
E__inference_conv1d_43_layer_call_and_return_conditional_losses_8666502#
!conv1d_43/StatefulPartitionedCall?
!conv1d_44/StatefulPartitionedCallStatefulPartitionedCall*conv1d_43/StatefulPartitionedCall:output:0conv1d_44_867023conv1d_44_867025*
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
E__inference_conv1d_44_layer_call_and_return_conditional_losses_8666842#
!conv1d_44/StatefulPartitionedCall?
!conv1d_45/StatefulPartitionedCallStatefulPartitionedCall*conv1d_44/StatefulPartitionedCall:output:0conv1d_45_867028conv1d_45_867030*
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
E__inference_conv1d_45_layer_call_and_return_conditional_losses_8667182#
!conv1d_45/StatefulPartitionedCall?
!conv1d_46/StatefulPartitionedCallStatefulPartitionedCall*conv1d_45/StatefulPartitionedCall:output:0conv1d_46_867033conv1d_46_867035*
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
E__inference_conv1d_46_layer_call_and_return_conditional_losses_8667522#
!conv1d_46/StatefulPartitionedCall?
!conv1d_47/StatefulPartitionedCallStatefulPartitionedCall*conv1d_46/StatefulPartitionedCall:output:0conv1d_47_867038conv1d_47_867040*
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
E__inference_conv1d_47_layer_call_and_return_conditional_losses_8667862#
!conv1d_47/StatefulPartitionedCall?
!conv1d_48/StatefulPartitionedCallStatefulPartitionedCall*conv1d_47/StatefulPartitionedCall:output:0conv1d_48_867043conv1d_48_867045*
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
E__inference_conv1d_48_layer_call_and_return_conditional_losses_8668202#
!conv1d_48/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_18_867048dense_18_867050*
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
GPU 2J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_8668472"
 dense_18/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall*conv1d_48/StatefulPartitionedCall:output:0*
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
E__inference_flatten_6_layer_call_and_return_conditional_losses_8668692
flatten_6/PartitionedCall?
concatenate_6/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0"flatten_6/PartitionedCall:output:0*
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
I__inference_concatenate_6_layer_call_and_return_conditional_losses_8668842
concatenate_6/PartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0dense_19_867055dense_19_867057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_8669042"
 dense_19/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_867060dense_20_867062*
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
GPU 2J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_8669312"
 dense_20/StatefulPartitionedCall?
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0"^conv1d_42/StatefulPartitionedCall"^conv1d_43/StatefulPartitionedCall"^conv1d_44/StatefulPartitionedCall"^conv1d_45/StatefulPartitionedCall"^conv1d_46/StatefulPartitionedCall"^conv1d_47/StatefulPartitionedCall"^conv1d_48/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2F
!conv1d_42/StatefulPartitionedCall!conv1d_42/StatefulPartitionedCall2F
!conv1d_43/StatefulPartitionedCall!conv1d_43/StatefulPartitionedCall2F
!conv1d_44/StatefulPartitionedCall!conv1d_44/StatefulPartitionedCall2F
!conv1d_45/StatefulPartitionedCall!conv1d_45/StatefulPartitionedCall2F
!conv1d_46/StatefulPartitionedCall!conv1d_46/StatefulPartitionedCall2F
!conv1d_47/StatefulPartitionedCall!conv1d_47/StatefulPartitionedCall2F
!conv1d_48/StatefulPartitionedCall!conv1d_48/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:S O
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
*__inference_conv1d_45_layer_call_fn_867724

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
E__inference_conv1d_45_layer_call_and_return_conditional_losses_8667182
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
E__inference_conv1d_48_layer_call_and_return_conditional_losses_867796

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
*__inference_conv1d_42_layer_call_fn_867643

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
E__inference_conv1d_42_layer_call_and_return_conditional_losses_8666162
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
E__inference_conv1d_47_layer_call_and_return_conditional_losses_866786

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
E__inference_conv1d_42_layer_call_and_return_conditional_losses_866616

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
?
$__inference_signature_wrapper_867268
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
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_8665932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
?
?
(__inference_model_6_layer_call_fn_867570
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
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_8670662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
?;
?
C__inference_model_6_layer_call_and_return_conditional_losses_866948
conv
cat
conv1d_42_866627
conv1d_42_866629
conv1d_43_866661
conv1d_43_866663
conv1d_44_866695
conv1d_44_866697
conv1d_45_866729
conv1d_45_866731
conv1d_46_866763
conv1d_46_866765
conv1d_47_866797
conv1d_47_866799
conv1d_48_866831
conv1d_48_866833
dense_18_866858
dense_18_866860
dense_19_866915
dense_19_866917
dense_20_866942
dense_20_866944
identity??!conv1d_42/StatefulPartitionedCall?!conv1d_43/StatefulPartitionedCall?!conv1d_44/StatefulPartitionedCall?!conv1d_45/StatefulPartitionedCall?!conv1d_46/StatefulPartitionedCall?!conv1d_47/StatefulPartitionedCall?!conv1d_48/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall?
!conv1d_42/StatefulPartitionedCallStatefulPartitionedCallconvconv1d_42_866627conv1d_42_866629*
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
E__inference_conv1d_42_layer_call_and_return_conditional_losses_8666162#
!conv1d_42/StatefulPartitionedCall?
!conv1d_43/StatefulPartitionedCallStatefulPartitionedCall*conv1d_42/StatefulPartitionedCall:output:0conv1d_43_866661conv1d_43_866663*
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
E__inference_conv1d_43_layer_call_and_return_conditional_losses_8666502#
!conv1d_43/StatefulPartitionedCall?
!conv1d_44/StatefulPartitionedCallStatefulPartitionedCall*conv1d_43/StatefulPartitionedCall:output:0conv1d_44_866695conv1d_44_866697*
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
E__inference_conv1d_44_layer_call_and_return_conditional_losses_8666842#
!conv1d_44/StatefulPartitionedCall?
!conv1d_45/StatefulPartitionedCallStatefulPartitionedCall*conv1d_44/StatefulPartitionedCall:output:0conv1d_45_866729conv1d_45_866731*
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
E__inference_conv1d_45_layer_call_and_return_conditional_losses_8667182#
!conv1d_45/StatefulPartitionedCall?
!conv1d_46/StatefulPartitionedCallStatefulPartitionedCall*conv1d_45/StatefulPartitionedCall:output:0conv1d_46_866763conv1d_46_866765*
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
E__inference_conv1d_46_layer_call_and_return_conditional_losses_8667522#
!conv1d_46/StatefulPartitionedCall?
!conv1d_47/StatefulPartitionedCallStatefulPartitionedCall*conv1d_46/StatefulPartitionedCall:output:0conv1d_47_866797conv1d_47_866799*
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
E__inference_conv1d_47_layer_call_and_return_conditional_losses_8667862#
!conv1d_47/StatefulPartitionedCall?
!conv1d_48/StatefulPartitionedCallStatefulPartitionedCall*conv1d_47/StatefulPartitionedCall:output:0conv1d_48_866831conv1d_48_866833*
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
E__inference_conv1d_48_layer_call_and_return_conditional_losses_8668202#
!conv1d_48/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCallcatdense_18_866858dense_18_866860*
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
GPU 2J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_8668472"
 dense_18/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall*conv1d_48/StatefulPartitionedCall:output:0*
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
E__inference_flatten_6_layer_call_and_return_conditional_losses_8668692
flatten_6/PartitionedCall?
concatenate_6/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0"flatten_6/PartitionedCall:output:0*
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
I__inference_concatenate_6_layer_call_and_return_conditional_losses_8668842
concatenate_6/PartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0dense_19_866915dense_19_866917*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_8669042"
 dense_19/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_866942dense_20_866944*
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
GPU 2J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_8669312"
 dense_20/StatefulPartitionedCall?
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0"^conv1d_42/StatefulPartitionedCall"^conv1d_43/StatefulPartitionedCall"^conv1d_44/StatefulPartitionedCall"^conv1d_45/StatefulPartitionedCall"^conv1d_46/StatefulPartitionedCall"^conv1d_47/StatefulPartitionedCall"^conv1d_48/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2F
!conv1d_42/StatefulPartitionedCall!conv1d_42/StatefulPartitionedCall2F
!conv1d_43/StatefulPartitionedCall!conv1d_43/StatefulPartitionedCall2F
!conv1d_44/StatefulPartitionedCall!conv1d_44/StatefulPartitionedCall2F
!conv1d_45/StatefulPartitionedCall!conv1d_45/StatefulPartitionedCall2F
!conv1d_46/StatefulPartitionedCall!conv1d_46/StatefulPartitionedCall2F
!conv1d_47/StatefulPartitionedCall!conv1d_47/StatefulPartitionedCall2F
!conv1d_48/StatefulPartitionedCall!conv1d_48/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:Q M
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
E__inference_conv1d_45_layer_call_and_return_conditional_losses_866718

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
D__inference_dense_20_layer_call_and_return_conditional_losses_866931

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
Z
.__inference_concatenate_6_layer_call_fn_867849
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
I__inference_concatenate_6_layer_call_and_return_conditional_losses_8668842
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:??????????:Q M
'
_output_shapes
:?????????
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
)__inference_dense_19_layer_call_fn_867869

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_8669042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
s
I__inference_concatenate_6_layer_call_and_return_conditional_losses_866884

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
':?????????:??????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?#
"__inference__traced_restore_868337
file_prefix%
!assignvariableop_conv1d_42_kernel%
!assignvariableop_1_conv1d_42_bias'
#assignvariableop_2_conv1d_43_kernel%
!assignvariableop_3_conv1d_43_bias'
#assignvariableop_4_conv1d_44_kernel%
!assignvariableop_5_conv1d_44_bias'
#assignvariableop_6_conv1d_45_kernel%
!assignvariableop_7_conv1d_45_bias'
#assignvariableop_8_conv1d_46_kernel%
!assignvariableop_9_conv1d_46_bias(
$assignvariableop_10_conv1d_47_kernel&
"assignvariableop_11_conv1d_47_bias(
$assignvariableop_12_conv1d_48_kernel&
"assignvariableop_13_conv1d_48_bias'
#assignvariableop_14_dense_18_kernel%
!assignvariableop_15_dense_18_bias'
#assignvariableop_16_dense_19_kernel%
!assignvariableop_17_dense_19_bias'
#assignvariableop_18_dense_20_kernel%
!assignvariableop_19_dense_20_bias!
assignvariableop_20_adam_iter#
assignvariableop_21_adam_beta_1#
assignvariableop_22_adam_beta_2"
assignvariableop_23_adam_decay*
&assignvariableop_24_adam_learning_rate
assignvariableop_25_total
assignvariableop_26_count
assignvariableop_27_total_1
assignvariableop_28_count_1/
+assignvariableop_29_adam_conv1d_42_kernel_m-
)assignvariableop_30_adam_conv1d_42_bias_m/
+assignvariableop_31_adam_conv1d_43_kernel_m-
)assignvariableop_32_adam_conv1d_43_bias_m/
+assignvariableop_33_adam_conv1d_44_kernel_m-
)assignvariableop_34_adam_conv1d_44_bias_m/
+assignvariableop_35_adam_conv1d_45_kernel_m-
)assignvariableop_36_adam_conv1d_45_bias_m/
+assignvariableop_37_adam_conv1d_46_kernel_m-
)assignvariableop_38_adam_conv1d_46_bias_m/
+assignvariableop_39_adam_conv1d_47_kernel_m-
)assignvariableop_40_adam_conv1d_47_bias_m/
+assignvariableop_41_adam_conv1d_48_kernel_m-
)assignvariableop_42_adam_conv1d_48_bias_m.
*assignvariableop_43_adam_dense_18_kernel_m,
(assignvariableop_44_adam_dense_18_bias_m.
*assignvariableop_45_adam_dense_19_kernel_m,
(assignvariableop_46_adam_dense_19_bias_m.
*assignvariableop_47_adam_dense_20_kernel_m,
(assignvariableop_48_adam_dense_20_bias_m/
+assignvariableop_49_adam_conv1d_42_kernel_v-
)assignvariableop_50_adam_conv1d_42_bias_v/
+assignvariableop_51_adam_conv1d_43_kernel_v-
)assignvariableop_52_adam_conv1d_43_bias_v/
+assignvariableop_53_adam_conv1d_44_kernel_v-
)assignvariableop_54_adam_conv1d_44_bias_v/
+assignvariableop_55_adam_conv1d_45_kernel_v-
)assignvariableop_56_adam_conv1d_45_bias_v/
+assignvariableop_57_adam_conv1d_46_kernel_v-
)assignvariableop_58_adam_conv1d_46_bias_v/
+assignvariableop_59_adam_conv1d_47_kernel_v-
)assignvariableop_60_adam_conv1d_47_bias_v/
+assignvariableop_61_adam_conv1d_48_kernel_v-
)assignvariableop_62_adam_conv1d_48_bias_v.
*assignvariableop_63_adam_dense_18_kernel_v,
(assignvariableop_64_adam_dense_18_bias_v.
*assignvariableop_65_adam_dense_19_kernel_v,
(assignvariableop_66_adam_dense_19_bias_v.
*assignvariableop_67_adam_dense_20_kernel_v,
(assignvariableop_68_adam_dense_20_bias_v
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
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_42_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_42_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv1d_43_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv1d_43_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv1d_44_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv1d_44_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv1d_45_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv1d_45_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv1d_46_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv1d_46_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv1d_47_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv1d_47_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv1d_48_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv1d_48_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_18_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_18_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_19_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_19_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_20_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_20_biasIdentity_19:output:0"/device:CPU:0*
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
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv1d_42_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv1d_42_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv1d_43_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv1d_43_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv1d_44_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv1d_44_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv1d_45_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv1d_45_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv1d_46_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv1d_46_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv1d_47_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv1d_47_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv1d_48_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv1d_48_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_18_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_18_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_19_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_19_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_20_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_20_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv1d_42_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv1d_42_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv1d_43_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv1d_43_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv1d_44_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv1d_44_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv1d_45_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv1d_45_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv1d_46_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv1d_46_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv1d_47_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv1d_47_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv1d_48_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv1d_48_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_18_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_18_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_19_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_19_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_20_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_20_bias_vIdentity_68:output:0"/device:CPU:0*
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
?

*__inference_conv1d_44_layer_call_fn_867697

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
E__inference_conv1d_44_layer_call_and_return_conditional_losses_8666842
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
)__inference_dense_20_layer_call_fn_867889

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
GPU 2J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_8669312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv1d_48_layer_call_and_return_conditional_losses_866820

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
C__inference_model_6_layer_call_and_return_conditional_losses_867524
inputs_0
inputs_19
5conv1d_42_conv1d_expanddims_1_readvariableop_resource-
)conv1d_42_biasadd_readvariableop_resource9
5conv1d_43_conv1d_expanddims_1_readvariableop_resource-
)conv1d_43_biasadd_readvariableop_resource9
5conv1d_44_conv1d_expanddims_1_readvariableop_resource-
)conv1d_44_biasadd_readvariableop_resource9
5conv1d_45_conv1d_expanddims_1_readvariableop_resource-
)conv1d_45_biasadd_readvariableop_resource9
5conv1d_46_conv1d_expanddims_1_readvariableop_resource-
)conv1d_46_biasadd_readvariableop_resource9
5conv1d_47_conv1d_expanddims_1_readvariableop_resource-
)conv1d_47_biasadd_readvariableop_resource9
5conv1d_48_conv1d_expanddims_1_readvariableop_resource-
)conv1d_48_biasadd_readvariableop_resource.
*dense_18_mlcmatmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource.
*dense_19_mlcmatmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource.
*dense_20_mlcmatmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource
identity?? conv1d_42/BiasAdd/ReadVariableOp?,conv1d_42/conv1d/ExpandDims_1/ReadVariableOp? conv1d_43/BiasAdd/ReadVariableOp?,conv1d_43/conv1d/ExpandDims_1/ReadVariableOp? conv1d_44/BiasAdd/ReadVariableOp?,conv1d_44/conv1d/ExpandDims_1/ReadVariableOp? conv1d_45/BiasAdd/ReadVariableOp?,conv1d_45/conv1d/ExpandDims_1/ReadVariableOp? conv1d_46/BiasAdd/ReadVariableOp?,conv1d_46/conv1d/ExpandDims_1/ReadVariableOp? conv1d_47/BiasAdd/ReadVariableOp?,conv1d_47/conv1d/ExpandDims_1/ReadVariableOp? conv1d_48/BiasAdd/ReadVariableOp?,conv1d_48/conv1d/ExpandDims_1/ReadVariableOp?dense_18/BiasAdd/ReadVariableOp?!dense_18/MLCMatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?!dense_19/MLCMatMul/ReadVariableOp?dense_20/BiasAdd/ReadVariableOp?!dense_20/MLCMatMul/ReadVariableOp?
conv1d_42/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_42/Pad/paddings?
conv1d_42/PadPadinputs_0conv1d_42/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
conv1d_42/Pad?
conv1d_42/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_42/conv1d/ExpandDims/dim?
conv1d_42/conv1d/ExpandDims
ExpandDimsconv1d_42/Pad:output:0(conv1d_42/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_42/conv1d/ExpandDims?
,conv1d_42/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_42_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_42/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_42/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_42/conv1d/ExpandDims_1/dim?
conv1d_42/conv1d/ExpandDims_1
ExpandDims4conv1d_42/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_42/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_42/conv1d/ExpandDims_1?
conv1d_42/conv1d	MLCConv2D$conv1d_42/conv1d/ExpandDims:output:0&conv1d_42/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_42/conv1d?
conv1d_42/conv1d/SqueezeSqueezeconv1d_42/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_42/conv1d/Squeeze?
 conv1d_42/BiasAdd/ReadVariableOpReadVariableOp)conv1d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_42/BiasAdd/ReadVariableOp?
conv1d_42/BiasAddBiasAdd!conv1d_42/conv1d/Squeeze:output:0(conv1d_42/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_42/BiasAddz
conv1d_42/ReluReluconv1d_42/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_42/Relu?
conv1d_43/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_43/Pad/paddings?
conv1d_43/PadPadconv1d_42/Relu:activations:0conv1d_43/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_43/Pad?
conv1d_43/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_43/conv1d/ExpandDims/dim?
conv1d_43/conv1d/ExpandDims
ExpandDimsconv1d_43/Pad:output:0(conv1d_43/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_43/conv1d/ExpandDims?
,conv1d_43/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_43_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_43/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_43/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_43/conv1d/ExpandDims_1/dim?
conv1d_43/conv1d/ExpandDims_1
ExpandDims4conv1d_43/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_43/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_43/conv1d/ExpandDims_1?
conv1d_43/conv1d	MLCConv2D$conv1d_43/conv1d/ExpandDims:output:0&conv1d_43/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_43/conv1d?
conv1d_43/conv1d/SqueezeSqueezeconv1d_43/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_43/conv1d/Squeeze?
 conv1d_43/BiasAdd/ReadVariableOpReadVariableOp)conv1d_43_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_43/BiasAdd/ReadVariableOp?
conv1d_43/BiasAddBiasAdd!conv1d_43/conv1d/Squeeze:output:0(conv1d_43/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_43/BiasAddz
conv1d_43/ReluReluconv1d_43/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_43/Relu?
conv1d_44/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_44/Pad/paddings?
conv1d_44/PadPadconv1d_43/Relu:activations:0conv1d_44/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_44/Pad?
conv1d_44/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_44/conv1d/ExpandDims/dim?
conv1d_44/conv1d/ExpandDims
ExpandDimsconv1d_44/Pad:output:0(conv1d_44/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_44/conv1d/ExpandDims?
,conv1d_44/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_44_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_44/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_44/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_44/conv1d/ExpandDims_1/dim?
conv1d_44/conv1d/ExpandDims_1
ExpandDims4conv1d_44/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_44/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_44/conv1d/ExpandDims_1?
conv1d_44/conv1d	MLCConv2D$conv1d_44/conv1d/ExpandDims:output:0&conv1d_44/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_44/conv1d?
conv1d_44/conv1d/SqueezeSqueezeconv1d_44/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_44/conv1d/Squeeze?
 conv1d_44/BiasAdd/ReadVariableOpReadVariableOp)conv1d_44_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_44/BiasAdd/ReadVariableOp?
conv1d_44/BiasAddBiasAdd!conv1d_44/conv1d/Squeeze:output:0(conv1d_44/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_44/BiasAddz
conv1d_44/ReluReluconv1d_44/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_44/Relu?
conv1d_45/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_45/Pad/paddings?
conv1d_45/PadPadconv1d_44/Relu:activations:0conv1d_45/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_45/Pad?
conv1d_45/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_45/conv1d/ExpandDims/dim?
conv1d_45/conv1d/ExpandDims
ExpandDimsconv1d_45/Pad:output:0(conv1d_45/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_45/conv1d/ExpandDims?
,conv1d_45/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_45_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_45/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_45/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_45/conv1d/ExpandDims_1/dim?
conv1d_45/conv1d/ExpandDims_1
ExpandDims4conv1d_45/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_45/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_45/conv1d/ExpandDims_1?
conv1d_45/conv1d	MLCConv2D$conv1d_45/conv1d/ExpandDims:output:0&conv1d_45/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_45/conv1d?
conv1d_45/conv1d/SqueezeSqueezeconv1d_45/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_45/conv1d/Squeeze?
 conv1d_45/BiasAdd/ReadVariableOpReadVariableOp)conv1d_45_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_45/BiasAdd/ReadVariableOp?
conv1d_45/BiasAddBiasAdd!conv1d_45/conv1d/Squeeze:output:0(conv1d_45/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_45/BiasAddz
conv1d_45/ReluReluconv1d_45/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_45/Relu?
conv1d_46/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_46/Pad/paddings?
conv1d_46/PadPadconv1d_45/Relu:activations:0conv1d_46/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_46/Pad?
conv1d_46/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_46/conv1d/ExpandDims/dim?
conv1d_46/conv1d/ExpandDims
ExpandDimsconv1d_46/Pad:output:0(conv1d_46/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_46/conv1d/ExpandDims?
,conv1d_46/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_46_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_46/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_46/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_46/conv1d/ExpandDims_1/dim?
conv1d_46/conv1d/ExpandDims_1
ExpandDims4conv1d_46/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_46/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_46/conv1d/ExpandDims_1?
conv1d_46/conv1d	MLCConv2D$conv1d_46/conv1d/ExpandDims:output:0&conv1d_46/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_46/conv1d?
conv1d_46/conv1d/SqueezeSqueezeconv1d_46/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_46/conv1d/Squeeze?
 conv1d_46/BiasAdd/ReadVariableOpReadVariableOp)conv1d_46_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_46/BiasAdd/ReadVariableOp?
conv1d_46/BiasAddBiasAdd!conv1d_46/conv1d/Squeeze:output:0(conv1d_46/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_46/BiasAddz
conv1d_46/ReluReluconv1d_46/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_46/Relu?
conv1d_47/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_47/Pad/paddings?
conv1d_47/PadPadconv1d_46/Relu:activations:0conv1d_47/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_47/Pad?
conv1d_47/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_47/conv1d/ExpandDims/dim?
conv1d_47/conv1d/ExpandDims
ExpandDimsconv1d_47/Pad:output:0(conv1d_47/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_47/conv1d/ExpandDims?
,conv1d_47/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_47_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_47/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_47/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_47/conv1d/ExpandDims_1/dim?
conv1d_47/conv1d/ExpandDims_1
ExpandDims4conv1d_47/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_47/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_47/conv1d/ExpandDims_1?
conv1d_47/conv1d	MLCConv2D$conv1d_47/conv1d/ExpandDims:output:0&conv1d_47/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_47/conv1d?
conv1d_47/conv1d/SqueezeSqueezeconv1d_47/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_47/conv1d/Squeeze?
 conv1d_47/BiasAdd/ReadVariableOpReadVariableOp)conv1d_47_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_47/BiasAdd/ReadVariableOp?
conv1d_47/BiasAddBiasAdd!conv1d_47/conv1d/Squeeze:output:0(conv1d_47/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_47/BiasAddz
conv1d_47/ReluReluconv1d_47/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_47/Relu?
conv1d_48/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_48/Pad/paddings?
conv1d_48/PadPadconv1d_47/Relu:activations:0conv1d_48/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_48/Pad?
conv1d_48/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_48/conv1d/ExpandDims/dim?
conv1d_48/conv1d/ExpandDims
ExpandDimsconv1d_48/Pad:output:0(conv1d_48/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_48/conv1d/ExpandDims?
,conv1d_48/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_48_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_48/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_48/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_48/conv1d/ExpandDims_1/dim?
conv1d_48/conv1d/ExpandDims_1
ExpandDims4conv1d_48/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_48/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_48/conv1d/ExpandDims_1?
conv1d_48/conv1d	MLCConv2D$conv1d_48/conv1d/ExpandDims:output:0&conv1d_48/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_48/conv1d?
conv1d_48/conv1d/SqueezeSqueezeconv1d_48/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_48/conv1d/Squeeze?
 conv1d_48/BiasAdd/ReadVariableOpReadVariableOp)conv1d_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_48/BiasAdd/ReadVariableOp?
conv1d_48/BiasAddBiasAdd!conv1d_48/conv1d/Squeeze:output:0(conv1d_48/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_48/BiasAddz
conv1d_48/ReluReluconv1d_48/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_48/Relu?
!dense_18/MLCMatMul/ReadVariableOpReadVariableOp*dense_18_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_18/MLCMatMul/ReadVariableOp?
dense_18/MLCMatMul	MLCMatMulinputs_1)dense_18/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_18/MLCMatMul?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_18/BiasAdd/ReadVariableOp?
dense_18/BiasAddBiasAdddense_18/MLCMatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_18/Relus
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_6/Const?
flatten_6/ReshapeReshapeconv1d_48/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_6/Reshapex
concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_6/concat/axis?
concatenate_6/concatConcatV2dense_18/Relu:activations:0flatten_6/Reshape:output:0"concatenate_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_6/concat?
!dense_19/MLCMatMul/ReadVariableOpReadVariableOp*dense_19_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_19/MLCMatMul/ReadVariableOp?
dense_19/MLCMatMul	MLCMatMulconcatenate_6/concat:output:0)dense_19/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/MLCMatMul?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp?
dense_19/BiasAddBiasAdddense_19/MLCMatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/BiasAdds
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_19/Relu?
!dense_20/MLCMatMul/ReadVariableOpReadVariableOp*dense_20_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_20/MLCMatMul/ReadVariableOp?
dense_20/MLCMatMul	MLCMatMuldense_19/Relu:activations:0)dense_20/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_20/MLCMatMul?
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp?
dense_20/BiasAddBiasAdddense_20/MLCMatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_20/BiasAdd|
dense_20/SoftmaxSoftmaxdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_20/Softmax?
IdentityIdentitydense_20/Softmax:softmax:0!^conv1d_42/BiasAdd/ReadVariableOp-^conv1d_42/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_43/BiasAdd/ReadVariableOp-^conv1d_43/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_44/BiasAdd/ReadVariableOp-^conv1d_44/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_45/BiasAdd/ReadVariableOp-^conv1d_45/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_46/BiasAdd/ReadVariableOp-^conv1d_46/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_47/BiasAdd/ReadVariableOp-^conv1d_47/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_48/BiasAdd/ReadVariableOp-^conv1d_48/conv1d/ExpandDims_1/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp"^dense_18/MLCMatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp"^dense_19/MLCMatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp"^dense_20/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2D
 conv1d_42/BiasAdd/ReadVariableOp conv1d_42/BiasAdd/ReadVariableOp2\
,conv1d_42/conv1d/ExpandDims_1/ReadVariableOp,conv1d_42/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_43/BiasAdd/ReadVariableOp conv1d_43/BiasAdd/ReadVariableOp2\
,conv1d_43/conv1d/ExpandDims_1/ReadVariableOp,conv1d_43/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_44/BiasAdd/ReadVariableOp conv1d_44/BiasAdd/ReadVariableOp2\
,conv1d_44/conv1d/ExpandDims_1/ReadVariableOp,conv1d_44/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_45/BiasAdd/ReadVariableOp conv1d_45/BiasAdd/ReadVariableOp2\
,conv1d_45/conv1d/ExpandDims_1/ReadVariableOp,conv1d_45/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_46/BiasAdd/ReadVariableOp conv1d_46/BiasAdd/ReadVariableOp2\
,conv1d_46/conv1d/ExpandDims_1/ReadVariableOp,conv1d_46/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_47/BiasAdd/ReadVariableOp conv1d_47/BiasAdd/ReadVariableOp2\
,conv1d_47/conv1d/ExpandDims_1/ReadVariableOp,conv1d_47/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_48/BiasAdd/ReadVariableOp conv1d_48/BiasAdd/ReadVariableOp2\
,conv1d_48/conv1d/ExpandDims_1/ReadVariableOp,conv1d_48/conv1d/ExpandDims_1/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2F
!dense_18/MLCMatMul/ReadVariableOp!dense_18/MLCMatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2F
!dense_19/MLCMatMul/ReadVariableOp!dense_19/MLCMatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2F
!dense_20/MLCMatMul/ReadVariableOp!dense_20/MLCMatMul/ReadVariableOp:U Q
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
D__inference_dense_19_layer_call_and_return_conditional_losses_866904

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
(__inference_model_6_layer_call_fn_867616
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
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_8671692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
??
?
__inference__traced_save_868120
file_prefix/
+savev2_conv1d_42_kernel_read_readvariableop-
)savev2_conv1d_42_bias_read_readvariableop/
+savev2_conv1d_43_kernel_read_readvariableop-
)savev2_conv1d_43_bias_read_readvariableop/
+savev2_conv1d_44_kernel_read_readvariableop-
)savev2_conv1d_44_bias_read_readvariableop/
+savev2_conv1d_45_kernel_read_readvariableop-
)savev2_conv1d_45_bias_read_readvariableop/
+savev2_conv1d_46_kernel_read_readvariableop-
)savev2_conv1d_46_bias_read_readvariableop/
+savev2_conv1d_47_kernel_read_readvariableop-
)savev2_conv1d_47_bias_read_readvariableop/
+savev2_conv1d_48_kernel_read_readvariableop-
)savev2_conv1d_48_bias_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv1d_42_kernel_m_read_readvariableop4
0savev2_adam_conv1d_42_bias_m_read_readvariableop6
2savev2_adam_conv1d_43_kernel_m_read_readvariableop4
0savev2_adam_conv1d_43_bias_m_read_readvariableop6
2savev2_adam_conv1d_44_kernel_m_read_readvariableop4
0savev2_adam_conv1d_44_bias_m_read_readvariableop6
2savev2_adam_conv1d_45_kernel_m_read_readvariableop4
0savev2_adam_conv1d_45_bias_m_read_readvariableop6
2savev2_adam_conv1d_46_kernel_m_read_readvariableop4
0savev2_adam_conv1d_46_bias_m_read_readvariableop6
2savev2_adam_conv1d_47_kernel_m_read_readvariableop4
0savev2_adam_conv1d_47_bias_m_read_readvariableop6
2savev2_adam_conv1d_48_kernel_m_read_readvariableop4
0savev2_adam_conv1d_48_bias_m_read_readvariableop5
1savev2_adam_dense_18_kernel_m_read_readvariableop3
/savev2_adam_dense_18_bias_m_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop3
/savev2_adam_dense_19_bias_m_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableop6
2savev2_adam_conv1d_42_kernel_v_read_readvariableop4
0savev2_adam_conv1d_42_bias_v_read_readvariableop6
2savev2_adam_conv1d_43_kernel_v_read_readvariableop4
0savev2_adam_conv1d_43_bias_v_read_readvariableop6
2savev2_adam_conv1d_44_kernel_v_read_readvariableop4
0savev2_adam_conv1d_44_bias_v_read_readvariableop6
2savev2_adam_conv1d_45_kernel_v_read_readvariableop4
0savev2_adam_conv1d_45_bias_v_read_readvariableop6
2savev2_adam_conv1d_46_kernel_v_read_readvariableop4
0savev2_adam_conv1d_46_bias_v_read_readvariableop6
2savev2_adam_conv1d_47_kernel_v_read_readvariableop4
0savev2_adam_conv1d_47_bias_v_read_readvariableop6
2savev2_adam_conv1d_48_kernel_v_read_readvariableop4
0savev2_adam_conv1d_48_bias_v_read_readvariableop5
1savev2_adam_dense_18_kernel_v_read_readvariableop3
/savev2_adam_dense_18_bias_v_read_readvariableop5
1savev2_adam_dense_19_kernel_v_read_readvariableop3
/savev2_adam_dense_19_bias_v_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv1d_42_kernel_read_readvariableop)savev2_conv1d_42_bias_read_readvariableop+savev2_conv1d_43_kernel_read_readvariableop)savev2_conv1d_43_bias_read_readvariableop+savev2_conv1d_44_kernel_read_readvariableop)savev2_conv1d_44_bias_read_readvariableop+savev2_conv1d_45_kernel_read_readvariableop)savev2_conv1d_45_bias_read_readvariableop+savev2_conv1d_46_kernel_read_readvariableop)savev2_conv1d_46_bias_read_readvariableop+savev2_conv1d_47_kernel_read_readvariableop)savev2_conv1d_47_bias_read_readvariableop+savev2_conv1d_48_kernel_read_readvariableop)savev2_conv1d_48_bias_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv1d_42_kernel_m_read_readvariableop0savev2_adam_conv1d_42_bias_m_read_readvariableop2savev2_adam_conv1d_43_kernel_m_read_readvariableop0savev2_adam_conv1d_43_bias_m_read_readvariableop2savev2_adam_conv1d_44_kernel_m_read_readvariableop0savev2_adam_conv1d_44_bias_m_read_readvariableop2savev2_adam_conv1d_45_kernel_m_read_readvariableop0savev2_adam_conv1d_45_bias_m_read_readvariableop2savev2_adam_conv1d_46_kernel_m_read_readvariableop0savev2_adam_conv1d_46_bias_m_read_readvariableop2savev2_adam_conv1d_47_kernel_m_read_readvariableop0savev2_adam_conv1d_47_bias_m_read_readvariableop2savev2_adam_conv1d_48_kernel_m_read_readvariableop0savev2_adam_conv1d_48_bias_m_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop/savev2_adam_dense_19_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop2savev2_adam_conv1d_42_kernel_v_read_readvariableop0savev2_adam_conv1d_42_bias_v_read_readvariableop2savev2_adam_conv1d_43_kernel_v_read_readvariableop0savev2_adam_conv1d_43_bias_v_read_readvariableop2savev2_adam_conv1d_44_kernel_v_read_readvariableop0savev2_adam_conv1d_44_bias_v_read_readvariableop2savev2_adam_conv1d_45_kernel_v_read_readvariableop0savev2_adam_conv1d_45_bias_v_read_readvariableop2savev2_adam_conv1d_46_kernel_v_read_readvariableop0savev2_adam_conv1d_46_bias_v_read_readvariableop2savev2_adam_conv1d_47_kernel_v_read_readvariableop0savev2_adam_conv1d_47_bias_v_read_readvariableop2savev2_adam_conv1d_48_kernel_v_read_readvariableop0savev2_adam_conv1d_48_bias_v_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop/savev2_adam_dense_19_bias_v_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:::	?:::: : : : : : : : : :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:::	?::::@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:::	?:::: 2(
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

:: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::
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

:: -

_output_shapes
::%.!

_output_shapes
:	?: /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::(2$
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

:: A

_output_shapes
::%B!

_output_shapes
:	?: C

_output_shapes
::$D 

_output_shapes

:: E

_output_shapes
::F

_output_shapes
: 
?
?
E__inference_conv1d_47_layer_call_and_return_conditional_losses_867769

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
E__inference_flatten_6_layer_call_and_return_conditional_losses_866869

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
?
(__inference_model_6_layer_call_fn_867109
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
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_8670662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
??
?
C__inference_model_6_layer_call_and_return_conditional_losses_867396
inputs_0
inputs_19
5conv1d_42_conv1d_expanddims_1_readvariableop_resource-
)conv1d_42_biasadd_readvariableop_resource9
5conv1d_43_conv1d_expanddims_1_readvariableop_resource-
)conv1d_43_biasadd_readvariableop_resource9
5conv1d_44_conv1d_expanddims_1_readvariableop_resource-
)conv1d_44_biasadd_readvariableop_resource9
5conv1d_45_conv1d_expanddims_1_readvariableop_resource-
)conv1d_45_biasadd_readvariableop_resource9
5conv1d_46_conv1d_expanddims_1_readvariableop_resource-
)conv1d_46_biasadd_readvariableop_resource9
5conv1d_47_conv1d_expanddims_1_readvariableop_resource-
)conv1d_47_biasadd_readvariableop_resource9
5conv1d_48_conv1d_expanddims_1_readvariableop_resource-
)conv1d_48_biasadd_readvariableop_resource.
*dense_18_mlcmatmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource.
*dense_19_mlcmatmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource.
*dense_20_mlcmatmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource
identity?? conv1d_42/BiasAdd/ReadVariableOp?,conv1d_42/conv1d/ExpandDims_1/ReadVariableOp? conv1d_43/BiasAdd/ReadVariableOp?,conv1d_43/conv1d/ExpandDims_1/ReadVariableOp? conv1d_44/BiasAdd/ReadVariableOp?,conv1d_44/conv1d/ExpandDims_1/ReadVariableOp? conv1d_45/BiasAdd/ReadVariableOp?,conv1d_45/conv1d/ExpandDims_1/ReadVariableOp? conv1d_46/BiasAdd/ReadVariableOp?,conv1d_46/conv1d/ExpandDims_1/ReadVariableOp? conv1d_47/BiasAdd/ReadVariableOp?,conv1d_47/conv1d/ExpandDims_1/ReadVariableOp? conv1d_48/BiasAdd/ReadVariableOp?,conv1d_48/conv1d/ExpandDims_1/ReadVariableOp?dense_18/BiasAdd/ReadVariableOp?!dense_18/MLCMatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?!dense_19/MLCMatMul/ReadVariableOp?dense_20/BiasAdd/ReadVariableOp?!dense_20/MLCMatMul/ReadVariableOp?
conv1d_42/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_42/Pad/paddings?
conv1d_42/PadPadinputs_0conv1d_42/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
conv1d_42/Pad?
conv1d_42/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_42/conv1d/ExpandDims/dim?
conv1d_42/conv1d/ExpandDims
ExpandDimsconv1d_42/Pad:output:0(conv1d_42/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_42/conv1d/ExpandDims?
,conv1d_42/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_42_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_42/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_42/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_42/conv1d/ExpandDims_1/dim?
conv1d_42/conv1d/ExpandDims_1
ExpandDims4conv1d_42/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_42/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_42/conv1d/ExpandDims_1?
conv1d_42/conv1d	MLCConv2D$conv1d_42/conv1d/ExpandDims:output:0&conv1d_42/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_42/conv1d?
conv1d_42/conv1d/SqueezeSqueezeconv1d_42/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_42/conv1d/Squeeze?
 conv1d_42/BiasAdd/ReadVariableOpReadVariableOp)conv1d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_42/BiasAdd/ReadVariableOp?
conv1d_42/BiasAddBiasAdd!conv1d_42/conv1d/Squeeze:output:0(conv1d_42/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_42/BiasAddz
conv1d_42/ReluReluconv1d_42/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_42/Relu?
conv1d_43/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_43/Pad/paddings?
conv1d_43/PadPadconv1d_42/Relu:activations:0conv1d_43/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_43/Pad?
conv1d_43/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_43/conv1d/ExpandDims/dim?
conv1d_43/conv1d/ExpandDims
ExpandDimsconv1d_43/Pad:output:0(conv1d_43/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_43/conv1d/ExpandDims?
,conv1d_43/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_43_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_43/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_43/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_43/conv1d/ExpandDims_1/dim?
conv1d_43/conv1d/ExpandDims_1
ExpandDims4conv1d_43/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_43/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_43/conv1d/ExpandDims_1?
conv1d_43/conv1d	MLCConv2D$conv1d_43/conv1d/ExpandDims:output:0&conv1d_43/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_43/conv1d?
conv1d_43/conv1d/SqueezeSqueezeconv1d_43/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_43/conv1d/Squeeze?
 conv1d_43/BiasAdd/ReadVariableOpReadVariableOp)conv1d_43_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_43/BiasAdd/ReadVariableOp?
conv1d_43/BiasAddBiasAdd!conv1d_43/conv1d/Squeeze:output:0(conv1d_43/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_43/BiasAddz
conv1d_43/ReluReluconv1d_43/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_43/Relu?
conv1d_44/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_44/Pad/paddings?
conv1d_44/PadPadconv1d_43/Relu:activations:0conv1d_44/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_44/Pad?
conv1d_44/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_44/conv1d/ExpandDims/dim?
conv1d_44/conv1d/ExpandDims
ExpandDimsconv1d_44/Pad:output:0(conv1d_44/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_44/conv1d/ExpandDims?
,conv1d_44/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_44_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_44/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_44/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_44/conv1d/ExpandDims_1/dim?
conv1d_44/conv1d/ExpandDims_1
ExpandDims4conv1d_44/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_44/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_44/conv1d/ExpandDims_1?
conv1d_44/conv1d	MLCConv2D$conv1d_44/conv1d/ExpandDims:output:0&conv1d_44/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_44/conv1d?
conv1d_44/conv1d/SqueezeSqueezeconv1d_44/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_44/conv1d/Squeeze?
 conv1d_44/BiasAdd/ReadVariableOpReadVariableOp)conv1d_44_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_44/BiasAdd/ReadVariableOp?
conv1d_44/BiasAddBiasAdd!conv1d_44/conv1d/Squeeze:output:0(conv1d_44/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_44/BiasAddz
conv1d_44/ReluReluconv1d_44/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_44/Relu?
conv1d_45/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_45/Pad/paddings?
conv1d_45/PadPadconv1d_44/Relu:activations:0conv1d_45/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_45/Pad?
conv1d_45/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_45/conv1d/ExpandDims/dim?
conv1d_45/conv1d/ExpandDims
ExpandDimsconv1d_45/Pad:output:0(conv1d_45/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_45/conv1d/ExpandDims?
,conv1d_45/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_45_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_45/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_45/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_45/conv1d/ExpandDims_1/dim?
conv1d_45/conv1d/ExpandDims_1
ExpandDims4conv1d_45/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_45/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_45/conv1d/ExpandDims_1?
conv1d_45/conv1d	MLCConv2D$conv1d_45/conv1d/ExpandDims:output:0&conv1d_45/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_45/conv1d?
conv1d_45/conv1d/SqueezeSqueezeconv1d_45/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_45/conv1d/Squeeze?
 conv1d_45/BiasAdd/ReadVariableOpReadVariableOp)conv1d_45_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_45/BiasAdd/ReadVariableOp?
conv1d_45/BiasAddBiasAdd!conv1d_45/conv1d/Squeeze:output:0(conv1d_45/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_45/BiasAddz
conv1d_45/ReluReluconv1d_45/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_45/Relu?
conv1d_46/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_46/Pad/paddings?
conv1d_46/PadPadconv1d_45/Relu:activations:0conv1d_46/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_46/Pad?
conv1d_46/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_46/conv1d/ExpandDims/dim?
conv1d_46/conv1d/ExpandDims
ExpandDimsconv1d_46/Pad:output:0(conv1d_46/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_46/conv1d/ExpandDims?
,conv1d_46/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_46_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_46/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_46/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_46/conv1d/ExpandDims_1/dim?
conv1d_46/conv1d/ExpandDims_1
ExpandDims4conv1d_46/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_46/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_46/conv1d/ExpandDims_1?
conv1d_46/conv1d	MLCConv2D$conv1d_46/conv1d/ExpandDims:output:0&conv1d_46/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_46/conv1d?
conv1d_46/conv1d/SqueezeSqueezeconv1d_46/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_46/conv1d/Squeeze?
 conv1d_46/BiasAdd/ReadVariableOpReadVariableOp)conv1d_46_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_46/BiasAdd/ReadVariableOp?
conv1d_46/BiasAddBiasAdd!conv1d_46/conv1d/Squeeze:output:0(conv1d_46/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_46/BiasAddz
conv1d_46/ReluReluconv1d_46/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_46/Relu?
conv1d_47/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_47/Pad/paddings?
conv1d_47/PadPadconv1d_46/Relu:activations:0conv1d_47/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_47/Pad?
conv1d_47/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_47/conv1d/ExpandDims/dim?
conv1d_47/conv1d/ExpandDims
ExpandDimsconv1d_47/Pad:output:0(conv1d_47/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_47/conv1d/ExpandDims?
,conv1d_47/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_47_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_47/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_47/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_47/conv1d/ExpandDims_1/dim?
conv1d_47/conv1d/ExpandDims_1
ExpandDims4conv1d_47/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_47/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_47/conv1d/ExpandDims_1?
conv1d_47/conv1d	MLCConv2D$conv1d_47/conv1d/ExpandDims:output:0&conv1d_47/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_47/conv1d?
conv1d_47/conv1d/SqueezeSqueezeconv1d_47/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_47/conv1d/Squeeze?
 conv1d_47/BiasAdd/ReadVariableOpReadVariableOp)conv1d_47_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_47/BiasAdd/ReadVariableOp?
conv1d_47/BiasAddBiasAdd!conv1d_47/conv1d/Squeeze:output:0(conv1d_47/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_47/BiasAddz
conv1d_47/ReluReluconv1d_47/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_47/Relu?
conv1d_48/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_48/Pad/paddings?
conv1d_48/PadPadconv1d_47/Relu:activations:0conv1d_48/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_48/Pad?
conv1d_48/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_48/conv1d/ExpandDims/dim?
conv1d_48/conv1d/ExpandDims
ExpandDimsconv1d_48/Pad:output:0(conv1d_48/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_48/conv1d/ExpandDims?
,conv1d_48/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_48_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_48/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_48/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_48/conv1d/ExpandDims_1/dim?
conv1d_48/conv1d/ExpandDims_1
ExpandDims4conv1d_48/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_48/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_48/conv1d/ExpandDims_1?
conv1d_48/conv1d	MLCConv2D$conv1d_48/conv1d/ExpandDims:output:0&conv1d_48/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv1d_48/conv1d?
conv1d_48/conv1d/SqueezeSqueezeconv1d_48/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_48/conv1d/Squeeze?
 conv1d_48/BiasAdd/ReadVariableOpReadVariableOp)conv1d_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_48/BiasAdd/ReadVariableOp?
conv1d_48/BiasAddBiasAdd!conv1d_48/conv1d/Squeeze:output:0(conv1d_48/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_48/BiasAddz
conv1d_48/ReluReluconv1d_48/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_48/Relu?
!dense_18/MLCMatMul/ReadVariableOpReadVariableOp*dense_18_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_18/MLCMatMul/ReadVariableOp?
dense_18/MLCMatMul	MLCMatMulinputs_1)dense_18/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_18/MLCMatMul?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_18/BiasAdd/ReadVariableOp?
dense_18/BiasAddBiasAdddense_18/MLCMatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_18/Relus
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_6/Const?
flatten_6/ReshapeReshapeconv1d_48/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_6/Reshapex
concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_6/concat/axis?
concatenate_6/concatConcatV2dense_18/Relu:activations:0flatten_6/Reshape:output:0"concatenate_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_6/concat?
!dense_19/MLCMatMul/ReadVariableOpReadVariableOp*dense_19_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_19/MLCMatMul/ReadVariableOp?
dense_19/MLCMatMul	MLCMatMulconcatenate_6/concat:output:0)dense_19/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/MLCMatMul?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp?
dense_19/BiasAddBiasAdddense_19/MLCMatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/BiasAdds
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_19/Relu?
!dense_20/MLCMatMul/ReadVariableOpReadVariableOp*dense_20_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_20/MLCMatMul/ReadVariableOp?
dense_20/MLCMatMul	MLCMatMuldense_19/Relu:activations:0)dense_20/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_20/MLCMatMul?
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp?
dense_20/BiasAddBiasAdddense_20/MLCMatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_20/BiasAdd|
dense_20/SoftmaxSoftmaxdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_20/Softmax?
IdentityIdentitydense_20/Softmax:softmax:0!^conv1d_42/BiasAdd/ReadVariableOp-^conv1d_42/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_43/BiasAdd/ReadVariableOp-^conv1d_43/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_44/BiasAdd/ReadVariableOp-^conv1d_44/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_45/BiasAdd/ReadVariableOp-^conv1d_45/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_46/BiasAdd/ReadVariableOp-^conv1d_46/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_47/BiasAdd/ReadVariableOp-^conv1d_47/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_48/BiasAdd/ReadVariableOp-^conv1d_48/conv1d/ExpandDims_1/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp"^dense_18/MLCMatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp"^dense_19/MLCMatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp"^dense_20/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2D
 conv1d_42/BiasAdd/ReadVariableOp conv1d_42/BiasAdd/ReadVariableOp2\
,conv1d_42/conv1d/ExpandDims_1/ReadVariableOp,conv1d_42/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_43/BiasAdd/ReadVariableOp conv1d_43/BiasAdd/ReadVariableOp2\
,conv1d_43/conv1d/ExpandDims_1/ReadVariableOp,conv1d_43/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_44/BiasAdd/ReadVariableOp conv1d_44/BiasAdd/ReadVariableOp2\
,conv1d_44/conv1d/ExpandDims_1/ReadVariableOp,conv1d_44/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_45/BiasAdd/ReadVariableOp conv1d_45/BiasAdd/ReadVariableOp2\
,conv1d_45/conv1d/ExpandDims_1/ReadVariableOp,conv1d_45/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_46/BiasAdd/ReadVariableOp conv1d_46/BiasAdd/ReadVariableOp2\
,conv1d_46/conv1d/ExpandDims_1/ReadVariableOp,conv1d_46/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_47/BiasAdd/ReadVariableOp conv1d_47/BiasAdd/ReadVariableOp2\
,conv1d_47/conv1d/ExpandDims_1/ReadVariableOp,conv1d_47/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_48/BiasAdd/ReadVariableOp conv1d_48/BiasAdd/ReadVariableOp2\
,conv1d_48/conv1d/ExpandDims_1/ReadVariableOp,conv1d_48/conv1d/ExpandDims_1/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2F
!dense_18/MLCMatMul/ReadVariableOp!dense_18/MLCMatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2F
!dense_19/MLCMatMul/ReadVariableOp!dense_19/MLCMatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2F
!dense_20/MLCMatMul/ReadVariableOp!dense_20/MLCMatMul/ReadVariableOp:U Q
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
??
?
!__inference__wrapped_model_866593
conv
catA
=model_6_conv1d_42_conv1d_expanddims_1_readvariableop_resource5
1model_6_conv1d_42_biasadd_readvariableop_resourceA
=model_6_conv1d_43_conv1d_expanddims_1_readvariableop_resource5
1model_6_conv1d_43_biasadd_readvariableop_resourceA
=model_6_conv1d_44_conv1d_expanddims_1_readvariableop_resource5
1model_6_conv1d_44_biasadd_readvariableop_resourceA
=model_6_conv1d_45_conv1d_expanddims_1_readvariableop_resource5
1model_6_conv1d_45_biasadd_readvariableop_resourceA
=model_6_conv1d_46_conv1d_expanddims_1_readvariableop_resource5
1model_6_conv1d_46_biasadd_readvariableop_resourceA
=model_6_conv1d_47_conv1d_expanddims_1_readvariableop_resource5
1model_6_conv1d_47_biasadd_readvariableop_resourceA
=model_6_conv1d_48_conv1d_expanddims_1_readvariableop_resource5
1model_6_conv1d_48_biasadd_readvariableop_resource6
2model_6_dense_18_mlcmatmul_readvariableop_resource4
0model_6_dense_18_biasadd_readvariableop_resource6
2model_6_dense_19_mlcmatmul_readvariableop_resource4
0model_6_dense_19_biasadd_readvariableop_resource6
2model_6_dense_20_mlcmatmul_readvariableop_resource4
0model_6_dense_20_biasadd_readvariableop_resource
identity??(model_6/conv1d_42/BiasAdd/ReadVariableOp?4model_6/conv1d_42/conv1d/ExpandDims_1/ReadVariableOp?(model_6/conv1d_43/BiasAdd/ReadVariableOp?4model_6/conv1d_43/conv1d/ExpandDims_1/ReadVariableOp?(model_6/conv1d_44/BiasAdd/ReadVariableOp?4model_6/conv1d_44/conv1d/ExpandDims_1/ReadVariableOp?(model_6/conv1d_45/BiasAdd/ReadVariableOp?4model_6/conv1d_45/conv1d/ExpandDims_1/ReadVariableOp?(model_6/conv1d_46/BiasAdd/ReadVariableOp?4model_6/conv1d_46/conv1d/ExpandDims_1/ReadVariableOp?(model_6/conv1d_47/BiasAdd/ReadVariableOp?4model_6/conv1d_47/conv1d/ExpandDims_1/ReadVariableOp?(model_6/conv1d_48/BiasAdd/ReadVariableOp?4model_6/conv1d_48/conv1d/ExpandDims_1/ReadVariableOp?'model_6/dense_18/BiasAdd/ReadVariableOp?)model_6/dense_18/MLCMatMul/ReadVariableOp?'model_6/dense_19/BiasAdd/ReadVariableOp?)model_6/dense_19/MLCMatMul/ReadVariableOp?'model_6/dense_20/BiasAdd/ReadVariableOp?)model_6/dense_20/MLCMatMul/ReadVariableOp?
model_6/conv1d_42/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_6/conv1d_42/Pad/paddings?
model_6/conv1d_42/PadPadconv'model_6/conv1d_42/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
model_6/conv1d_42/Pad?
'model_6/conv1d_42/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_6/conv1d_42/conv1d/ExpandDims/dim?
#model_6/conv1d_42/conv1d/ExpandDims
ExpandDimsmodel_6/conv1d_42/Pad:output:00model_6/conv1d_42/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2%
#model_6/conv1d_42/conv1d/ExpandDims?
4model_6/conv1d_42/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_6_conv1d_42_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype026
4model_6/conv1d_42/conv1d/ExpandDims_1/ReadVariableOp?
)model_6/conv1d_42/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_6/conv1d_42/conv1d/ExpandDims_1/dim?
%model_6/conv1d_42/conv1d/ExpandDims_1
ExpandDims<model_6/conv1d_42/conv1d/ExpandDims_1/ReadVariableOp:value:02model_6/conv1d_42/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2'
%model_6/conv1d_42/conv1d/ExpandDims_1?
model_6/conv1d_42/conv1d	MLCConv2D,model_6/conv1d_42/conv1d/ExpandDims:output:0.model_6/conv1d_42/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_6/conv1d_42/conv1d?
 model_6/conv1d_42/conv1d/SqueezeSqueeze!model_6/conv1d_42/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_6/conv1d_42/conv1d/Squeeze?
(model_6/conv1d_42/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv1d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_6/conv1d_42/BiasAdd/ReadVariableOp?
model_6/conv1d_42/BiasAddBiasAdd)model_6/conv1d_42/conv1d/Squeeze:output:00model_6/conv1d_42/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_42/BiasAdd?
model_6/conv1d_42/ReluRelu"model_6/conv1d_42/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_42/Relu?
model_6/conv1d_43/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_6/conv1d_43/Pad/paddings?
model_6/conv1d_43/PadPad$model_6/conv1d_42/Relu:activations:0'model_6/conv1d_43/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_43/Pad?
'model_6/conv1d_43/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_6/conv1d_43/conv1d/ExpandDims/dim?
#model_6/conv1d_43/conv1d/ExpandDims
ExpandDimsmodel_6/conv1d_43/Pad:output:00model_6/conv1d_43/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_6/conv1d_43/conv1d/ExpandDims?
4model_6/conv1d_43/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_6_conv1d_43_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_6/conv1d_43/conv1d/ExpandDims_1/ReadVariableOp?
)model_6/conv1d_43/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_6/conv1d_43/conv1d/ExpandDims_1/dim?
%model_6/conv1d_43/conv1d/ExpandDims_1
ExpandDims<model_6/conv1d_43/conv1d/ExpandDims_1/ReadVariableOp:value:02model_6/conv1d_43/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_6/conv1d_43/conv1d/ExpandDims_1?
model_6/conv1d_43/conv1d	MLCConv2D,model_6/conv1d_43/conv1d/ExpandDims:output:0.model_6/conv1d_43/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_6/conv1d_43/conv1d?
 model_6/conv1d_43/conv1d/SqueezeSqueeze!model_6/conv1d_43/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_6/conv1d_43/conv1d/Squeeze?
(model_6/conv1d_43/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv1d_43_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_6/conv1d_43/BiasAdd/ReadVariableOp?
model_6/conv1d_43/BiasAddBiasAdd)model_6/conv1d_43/conv1d/Squeeze:output:00model_6/conv1d_43/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_43/BiasAdd?
model_6/conv1d_43/ReluRelu"model_6/conv1d_43/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_43/Relu?
model_6/conv1d_44/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_6/conv1d_44/Pad/paddings?
model_6/conv1d_44/PadPad$model_6/conv1d_43/Relu:activations:0'model_6/conv1d_44/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_44/Pad?
'model_6/conv1d_44/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_6/conv1d_44/conv1d/ExpandDims/dim?
#model_6/conv1d_44/conv1d/ExpandDims
ExpandDimsmodel_6/conv1d_44/Pad:output:00model_6/conv1d_44/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_6/conv1d_44/conv1d/ExpandDims?
4model_6/conv1d_44/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_6_conv1d_44_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_6/conv1d_44/conv1d/ExpandDims_1/ReadVariableOp?
)model_6/conv1d_44/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_6/conv1d_44/conv1d/ExpandDims_1/dim?
%model_6/conv1d_44/conv1d/ExpandDims_1
ExpandDims<model_6/conv1d_44/conv1d/ExpandDims_1/ReadVariableOp:value:02model_6/conv1d_44/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_6/conv1d_44/conv1d/ExpandDims_1?
model_6/conv1d_44/conv1d	MLCConv2D,model_6/conv1d_44/conv1d/ExpandDims:output:0.model_6/conv1d_44/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_6/conv1d_44/conv1d?
 model_6/conv1d_44/conv1d/SqueezeSqueeze!model_6/conv1d_44/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_6/conv1d_44/conv1d/Squeeze?
(model_6/conv1d_44/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv1d_44_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_6/conv1d_44/BiasAdd/ReadVariableOp?
model_6/conv1d_44/BiasAddBiasAdd)model_6/conv1d_44/conv1d/Squeeze:output:00model_6/conv1d_44/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_44/BiasAdd?
model_6/conv1d_44/ReluRelu"model_6/conv1d_44/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_44/Relu?
model_6/conv1d_45/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_6/conv1d_45/Pad/paddings?
model_6/conv1d_45/PadPad$model_6/conv1d_44/Relu:activations:0'model_6/conv1d_45/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_45/Pad?
'model_6/conv1d_45/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_6/conv1d_45/conv1d/ExpandDims/dim?
#model_6/conv1d_45/conv1d/ExpandDims
ExpandDimsmodel_6/conv1d_45/Pad:output:00model_6/conv1d_45/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_6/conv1d_45/conv1d/ExpandDims?
4model_6/conv1d_45/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_6_conv1d_45_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_6/conv1d_45/conv1d/ExpandDims_1/ReadVariableOp?
)model_6/conv1d_45/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_6/conv1d_45/conv1d/ExpandDims_1/dim?
%model_6/conv1d_45/conv1d/ExpandDims_1
ExpandDims<model_6/conv1d_45/conv1d/ExpandDims_1/ReadVariableOp:value:02model_6/conv1d_45/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_6/conv1d_45/conv1d/ExpandDims_1?
model_6/conv1d_45/conv1d	MLCConv2D,model_6/conv1d_45/conv1d/ExpandDims:output:0.model_6/conv1d_45/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_6/conv1d_45/conv1d?
 model_6/conv1d_45/conv1d/SqueezeSqueeze!model_6/conv1d_45/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_6/conv1d_45/conv1d/Squeeze?
(model_6/conv1d_45/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv1d_45_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_6/conv1d_45/BiasAdd/ReadVariableOp?
model_6/conv1d_45/BiasAddBiasAdd)model_6/conv1d_45/conv1d/Squeeze:output:00model_6/conv1d_45/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_45/BiasAdd?
model_6/conv1d_45/ReluRelu"model_6/conv1d_45/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_45/Relu?
model_6/conv1d_46/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_6/conv1d_46/Pad/paddings?
model_6/conv1d_46/PadPad$model_6/conv1d_45/Relu:activations:0'model_6/conv1d_46/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_46/Pad?
'model_6/conv1d_46/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_6/conv1d_46/conv1d/ExpandDims/dim?
#model_6/conv1d_46/conv1d/ExpandDims
ExpandDimsmodel_6/conv1d_46/Pad:output:00model_6/conv1d_46/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_6/conv1d_46/conv1d/ExpandDims?
4model_6/conv1d_46/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_6_conv1d_46_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_6/conv1d_46/conv1d/ExpandDims_1/ReadVariableOp?
)model_6/conv1d_46/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_6/conv1d_46/conv1d/ExpandDims_1/dim?
%model_6/conv1d_46/conv1d/ExpandDims_1
ExpandDims<model_6/conv1d_46/conv1d/ExpandDims_1/ReadVariableOp:value:02model_6/conv1d_46/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_6/conv1d_46/conv1d/ExpandDims_1?
model_6/conv1d_46/conv1d	MLCConv2D,model_6/conv1d_46/conv1d/ExpandDims:output:0.model_6/conv1d_46/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_6/conv1d_46/conv1d?
 model_6/conv1d_46/conv1d/SqueezeSqueeze!model_6/conv1d_46/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_6/conv1d_46/conv1d/Squeeze?
(model_6/conv1d_46/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv1d_46_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_6/conv1d_46/BiasAdd/ReadVariableOp?
model_6/conv1d_46/BiasAddBiasAdd)model_6/conv1d_46/conv1d/Squeeze:output:00model_6/conv1d_46/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_46/BiasAdd?
model_6/conv1d_46/ReluRelu"model_6/conv1d_46/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_46/Relu?
model_6/conv1d_47/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_6/conv1d_47/Pad/paddings?
model_6/conv1d_47/PadPad$model_6/conv1d_46/Relu:activations:0'model_6/conv1d_47/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_47/Pad?
'model_6/conv1d_47/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_6/conv1d_47/conv1d/ExpandDims/dim?
#model_6/conv1d_47/conv1d/ExpandDims
ExpandDimsmodel_6/conv1d_47/Pad:output:00model_6/conv1d_47/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_6/conv1d_47/conv1d/ExpandDims?
4model_6/conv1d_47/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_6_conv1d_47_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_6/conv1d_47/conv1d/ExpandDims_1/ReadVariableOp?
)model_6/conv1d_47/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_6/conv1d_47/conv1d/ExpandDims_1/dim?
%model_6/conv1d_47/conv1d/ExpandDims_1
ExpandDims<model_6/conv1d_47/conv1d/ExpandDims_1/ReadVariableOp:value:02model_6/conv1d_47/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_6/conv1d_47/conv1d/ExpandDims_1?
model_6/conv1d_47/conv1d	MLCConv2D,model_6/conv1d_47/conv1d/ExpandDims:output:0.model_6/conv1d_47/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_6/conv1d_47/conv1d?
 model_6/conv1d_47/conv1d/SqueezeSqueeze!model_6/conv1d_47/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_6/conv1d_47/conv1d/Squeeze?
(model_6/conv1d_47/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv1d_47_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_6/conv1d_47/BiasAdd/ReadVariableOp?
model_6/conv1d_47/BiasAddBiasAdd)model_6/conv1d_47/conv1d/Squeeze:output:00model_6/conv1d_47/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_47/BiasAdd?
model_6/conv1d_47/ReluRelu"model_6/conv1d_47/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_47/Relu?
model_6/conv1d_48/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2 
model_6/conv1d_48/Pad/paddings?
model_6/conv1d_48/PadPad$model_6/conv1d_47/Relu:activations:0'model_6/conv1d_48/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_48/Pad?
'model_6/conv1d_48/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_6/conv1d_48/conv1d/ExpandDims/dim?
#model_6/conv1d_48/conv1d/ExpandDims
ExpandDimsmodel_6/conv1d_48/Pad:output:00model_6/conv1d_48/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#model_6/conv1d_48/conv1d/ExpandDims?
4model_6/conv1d_48/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_6_conv1d_48_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_6/conv1d_48/conv1d/ExpandDims_1/ReadVariableOp?
)model_6/conv1d_48/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_6/conv1d_48/conv1d/ExpandDims_1/dim?
%model_6/conv1d_48/conv1d/ExpandDims_1
ExpandDims<model_6/conv1d_48/conv1d/ExpandDims_1/ReadVariableOp:value:02model_6/conv1d_48/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2'
%model_6/conv1d_48/conv1d/ExpandDims_1?
model_6/conv1d_48/conv1d	MLCConv2D,model_6/conv1d_48/conv1d/ExpandDims:output:0.model_6/conv1d_48/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
model_6/conv1d_48/conv1d?
 model_6/conv1d_48/conv1d/SqueezeSqueeze!model_6/conv1d_48/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2"
 model_6/conv1d_48/conv1d/Squeeze?
(model_6/conv1d_48/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv1d_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_6/conv1d_48/BiasAdd/ReadVariableOp?
model_6/conv1d_48/BiasAddBiasAdd)model_6/conv1d_48/conv1d/Squeeze:output:00model_6/conv1d_48/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_48/BiasAdd?
model_6/conv1d_48/ReluRelu"model_6/conv1d_48/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model_6/conv1d_48/Relu?
)model_6/dense_18/MLCMatMul/ReadVariableOpReadVariableOp2model_6_dense_18_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)model_6/dense_18/MLCMatMul/ReadVariableOp?
model_6/dense_18/MLCMatMul	MLCMatMulcat1model_6/dense_18/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/dense_18/MLCMatMul?
'model_6/dense_18/BiasAdd/ReadVariableOpReadVariableOp0model_6_dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_6/dense_18/BiasAdd/ReadVariableOp?
model_6/dense_18/BiasAddBiasAdd$model_6/dense_18/MLCMatMul:product:0/model_6/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/dense_18/BiasAdd?
model_6/dense_18/ReluRelu!model_6/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_6/dense_18/Relu?
model_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model_6/flatten_6/Const?
model_6/flatten_6/ReshapeReshape$model_6/conv1d_48/Relu:activations:0 model_6/flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????2
model_6/flatten_6/Reshape?
!model_6/concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_6/concatenate_6/concat/axis?
model_6/concatenate_6/concatConcatV2#model_6/dense_18/Relu:activations:0"model_6/flatten_6/Reshape:output:0*model_6/concatenate_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model_6/concatenate_6/concat?
)model_6/dense_19/MLCMatMul/ReadVariableOpReadVariableOp2model_6_dense_19_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)model_6/dense_19/MLCMatMul/ReadVariableOp?
model_6/dense_19/MLCMatMul	MLCMatMul%model_6/concatenate_6/concat:output:01model_6/dense_19/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/dense_19/MLCMatMul?
'model_6/dense_19/BiasAdd/ReadVariableOpReadVariableOp0model_6_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_6/dense_19/BiasAdd/ReadVariableOp?
model_6/dense_19/BiasAddBiasAdd$model_6/dense_19/MLCMatMul:product:0/model_6/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/dense_19/BiasAdd?
model_6/dense_19/ReluRelu!model_6/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_6/dense_19/Relu?
)model_6/dense_20/MLCMatMul/ReadVariableOpReadVariableOp2model_6_dense_20_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)model_6/dense_20/MLCMatMul/ReadVariableOp?
model_6/dense_20/MLCMatMul	MLCMatMul#model_6/dense_19/Relu:activations:01model_6/dense_20/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/dense_20/MLCMatMul?
'model_6/dense_20/BiasAdd/ReadVariableOpReadVariableOp0model_6_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_6/dense_20/BiasAdd/ReadVariableOp?
model_6/dense_20/BiasAddBiasAdd$model_6/dense_20/MLCMatMul:product:0/model_6/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/dense_20/BiasAdd?
model_6/dense_20/SoftmaxSoftmax!model_6/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_6/dense_20/Softmax?
IdentityIdentity"model_6/dense_20/Softmax:softmax:0)^model_6/conv1d_42/BiasAdd/ReadVariableOp5^model_6/conv1d_42/conv1d/ExpandDims_1/ReadVariableOp)^model_6/conv1d_43/BiasAdd/ReadVariableOp5^model_6/conv1d_43/conv1d/ExpandDims_1/ReadVariableOp)^model_6/conv1d_44/BiasAdd/ReadVariableOp5^model_6/conv1d_44/conv1d/ExpandDims_1/ReadVariableOp)^model_6/conv1d_45/BiasAdd/ReadVariableOp5^model_6/conv1d_45/conv1d/ExpandDims_1/ReadVariableOp)^model_6/conv1d_46/BiasAdd/ReadVariableOp5^model_6/conv1d_46/conv1d/ExpandDims_1/ReadVariableOp)^model_6/conv1d_47/BiasAdd/ReadVariableOp5^model_6/conv1d_47/conv1d/ExpandDims_1/ReadVariableOp)^model_6/conv1d_48/BiasAdd/ReadVariableOp5^model_6/conv1d_48/conv1d/ExpandDims_1/ReadVariableOp(^model_6/dense_18/BiasAdd/ReadVariableOp*^model_6/dense_18/MLCMatMul/ReadVariableOp(^model_6/dense_19/BiasAdd/ReadVariableOp*^model_6/dense_19/MLCMatMul/ReadVariableOp(^model_6/dense_20/BiasAdd/ReadVariableOp*^model_6/dense_20/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2T
(model_6/conv1d_42/BiasAdd/ReadVariableOp(model_6/conv1d_42/BiasAdd/ReadVariableOp2l
4model_6/conv1d_42/conv1d/ExpandDims_1/ReadVariableOp4model_6/conv1d_42/conv1d/ExpandDims_1/ReadVariableOp2T
(model_6/conv1d_43/BiasAdd/ReadVariableOp(model_6/conv1d_43/BiasAdd/ReadVariableOp2l
4model_6/conv1d_43/conv1d/ExpandDims_1/ReadVariableOp4model_6/conv1d_43/conv1d/ExpandDims_1/ReadVariableOp2T
(model_6/conv1d_44/BiasAdd/ReadVariableOp(model_6/conv1d_44/BiasAdd/ReadVariableOp2l
4model_6/conv1d_44/conv1d/ExpandDims_1/ReadVariableOp4model_6/conv1d_44/conv1d/ExpandDims_1/ReadVariableOp2T
(model_6/conv1d_45/BiasAdd/ReadVariableOp(model_6/conv1d_45/BiasAdd/ReadVariableOp2l
4model_6/conv1d_45/conv1d/ExpandDims_1/ReadVariableOp4model_6/conv1d_45/conv1d/ExpandDims_1/ReadVariableOp2T
(model_6/conv1d_46/BiasAdd/ReadVariableOp(model_6/conv1d_46/BiasAdd/ReadVariableOp2l
4model_6/conv1d_46/conv1d/ExpandDims_1/ReadVariableOp4model_6/conv1d_46/conv1d/ExpandDims_1/ReadVariableOp2T
(model_6/conv1d_47/BiasAdd/ReadVariableOp(model_6/conv1d_47/BiasAdd/ReadVariableOp2l
4model_6/conv1d_47/conv1d/ExpandDims_1/ReadVariableOp4model_6/conv1d_47/conv1d/ExpandDims_1/ReadVariableOp2T
(model_6/conv1d_48/BiasAdd/ReadVariableOp(model_6/conv1d_48/BiasAdd/ReadVariableOp2l
4model_6/conv1d_48/conv1d/ExpandDims_1/ReadVariableOp4model_6/conv1d_48/conv1d/ExpandDims_1/ReadVariableOp2R
'model_6/dense_18/BiasAdd/ReadVariableOp'model_6/dense_18/BiasAdd/ReadVariableOp2V
)model_6/dense_18/MLCMatMul/ReadVariableOp)model_6/dense_18/MLCMatMul/ReadVariableOp2R
'model_6/dense_19/BiasAdd/ReadVariableOp'model_6/dense_19/BiasAdd/ReadVariableOp2V
)model_6/dense_19/MLCMatMul/ReadVariableOp)model_6/dense_19/MLCMatMul/ReadVariableOp2R
'model_6/dense_20/BiasAdd/ReadVariableOp'model_6/dense_20/BiasAdd/ReadVariableOp2V
)model_6/dense_20/MLCMatMul/ReadVariableOp)model_6/dense_20/MLCMatMul/ReadVariableOp:Q M
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
E__inference_conv1d_43_layer_call_and_return_conditional_losses_867661

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
F
*__inference_flatten_6_layer_call_fn_867836

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
E__inference_flatten_6_layer_call_and_return_conditional_losses_8668692
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

?
D__inference_dense_18_layer_call_and_return_conditional_losses_867816

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
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
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?;
?
C__inference_model_6_layer_call_and_return_conditional_losses_867169

inputs
inputs_1
conv1d_42_867116
conv1d_42_867118
conv1d_43_867121
conv1d_43_867123
conv1d_44_867126
conv1d_44_867128
conv1d_45_867131
conv1d_45_867133
conv1d_46_867136
conv1d_46_867138
conv1d_47_867141
conv1d_47_867143
conv1d_48_867146
conv1d_48_867148
dense_18_867151
dense_18_867153
dense_19_867158
dense_19_867160
dense_20_867163
dense_20_867165
identity??!conv1d_42/StatefulPartitionedCall?!conv1d_43/StatefulPartitionedCall?!conv1d_44/StatefulPartitionedCall?!conv1d_45/StatefulPartitionedCall?!conv1d_46/StatefulPartitionedCall?!conv1d_47/StatefulPartitionedCall?!conv1d_48/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall?
!conv1d_42/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_42_867116conv1d_42_867118*
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
E__inference_conv1d_42_layer_call_and_return_conditional_losses_8666162#
!conv1d_42/StatefulPartitionedCall?
!conv1d_43/StatefulPartitionedCallStatefulPartitionedCall*conv1d_42/StatefulPartitionedCall:output:0conv1d_43_867121conv1d_43_867123*
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
E__inference_conv1d_43_layer_call_and_return_conditional_losses_8666502#
!conv1d_43/StatefulPartitionedCall?
!conv1d_44/StatefulPartitionedCallStatefulPartitionedCall*conv1d_43/StatefulPartitionedCall:output:0conv1d_44_867126conv1d_44_867128*
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
E__inference_conv1d_44_layer_call_and_return_conditional_losses_8666842#
!conv1d_44/StatefulPartitionedCall?
!conv1d_45/StatefulPartitionedCallStatefulPartitionedCall*conv1d_44/StatefulPartitionedCall:output:0conv1d_45_867131conv1d_45_867133*
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
E__inference_conv1d_45_layer_call_and_return_conditional_losses_8667182#
!conv1d_45/StatefulPartitionedCall?
!conv1d_46/StatefulPartitionedCallStatefulPartitionedCall*conv1d_45/StatefulPartitionedCall:output:0conv1d_46_867136conv1d_46_867138*
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
E__inference_conv1d_46_layer_call_and_return_conditional_losses_8667522#
!conv1d_46/StatefulPartitionedCall?
!conv1d_47/StatefulPartitionedCallStatefulPartitionedCall*conv1d_46/StatefulPartitionedCall:output:0conv1d_47_867141conv1d_47_867143*
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
E__inference_conv1d_47_layer_call_and_return_conditional_losses_8667862#
!conv1d_47/StatefulPartitionedCall?
!conv1d_48/StatefulPartitionedCallStatefulPartitionedCall*conv1d_47/StatefulPartitionedCall:output:0conv1d_48_867146conv1d_48_867148*
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
E__inference_conv1d_48_layer_call_and_return_conditional_losses_8668202#
!conv1d_48/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_18_867151dense_18_867153*
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
GPU 2J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_8668472"
 dense_18/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall*conv1d_48/StatefulPartitionedCall:output:0*
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
E__inference_flatten_6_layer_call_and_return_conditional_losses_8668692
flatten_6/PartitionedCall?
concatenate_6/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0"flatten_6/PartitionedCall:output:0*
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
I__inference_concatenate_6_layer_call_and_return_conditional_losses_8668842
concatenate_6/PartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0dense_19_867158dense_19_867160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_8669042"
 dense_19/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_867163dense_20_867165*
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
GPU 2J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_8669312"
 dense_20/StatefulPartitionedCall?
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0"^conv1d_42/StatefulPartitionedCall"^conv1d_43/StatefulPartitionedCall"^conv1d_44/StatefulPartitionedCall"^conv1d_45/StatefulPartitionedCall"^conv1d_46/StatefulPartitionedCall"^conv1d_47/StatefulPartitionedCall"^conv1d_48/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2F
!conv1d_42/StatefulPartitionedCall!conv1d_42/StatefulPartitionedCall2F
!conv1d_43/StatefulPartitionedCall!conv1d_43/StatefulPartitionedCall2F
!conv1d_44/StatefulPartitionedCall!conv1d_44/StatefulPartitionedCall2F
!conv1d_45/StatefulPartitionedCall!conv1d_45/StatefulPartitionedCall2F
!conv1d_46/StatefulPartitionedCall!conv1d_46/StatefulPartitionedCall2F
!conv1d_47/StatefulPartitionedCall!conv1d_47/StatefulPartitionedCall2F
!conv1d_48/StatefulPartitionedCall!conv1d_48/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv1d_44_layer_call_and_return_conditional_losses_866684

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
*__inference_conv1d_48_layer_call_fn_867805

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
E__inference_conv1d_48_layer_call_and_return_conditional_losses_8668202
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
?
u
I__inference_concatenate_6_layer_call_and_return_conditional_losses_867843
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
':?????????:??????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?;
?
C__inference_model_6_layer_call_and_return_conditional_losses_867005
conv
cat
conv1d_42_866952
conv1d_42_866954
conv1d_43_866957
conv1d_43_866959
conv1d_44_866962
conv1d_44_866964
conv1d_45_866967
conv1d_45_866969
conv1d_46_866972
conv1d_46_866974
conv1d_47_866977
conv1d_47_866979
conv1d_48_866982
conv1d_48_866984
dense_18_866987
dense_18_866989
dense_19_866994
dense_19_866996
dense_20_866999
dense_20_867001
identity??!conv1d_42/StatefulPartitionedCall?!conv1d_43/StatefulPartitionedCall?!conv1d_44/StatefulPartitionedCall?!conv1d_45/StatefulPartitionedCall?!conv1d_46/StatefulPartitionedCall?!conv1d_47/StatefulPartitionedCall?!conv1d_48/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall?
!conv1d_42/StatefulPartitionedCallStatefulPartitionedCallconvconv1d_42_866952conv1d_42_866954*
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
E__inference_conv1d_42_layer_call_and_return_conditional_losses_8666162#
!conv1d_42/StatefulPartitionedCall?
!conv1d_43/StatefulPartitionedCallStatefulPartitionedCall*conv1d_42/StatefulPartitionedCall:output:0conv1d_43_866957conv1d_43_866959*
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
E__inference_conv1d_43_layer_call_and_return_conditional_losses_8666502#
!conv1d_43/StatefulPartitionedCall?
!conv1d_44/StatefulPartitionedCallStatefulPartitionedCall*conv1d_43/StatefulPartitionedCall:output:0conv1d_44_866962conv1d_44_866964*
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
E__inference_conv1d_44_layer_call_and_return_conditional_losses_8666842#
!conv1d_44/StatefulPartitionedCall?
!conv1d_45/StatefulPartitionedCallStatefulPartitionedCall*conv1d_44/StatefulPartitionedCall:output:0conv1d_45_866967conv1d_45_866969*
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
E__inference_conv1d_45_layer_call_and_return_conditional_losses_8667182#
!conv1d_45/StatefulPartitionedCall?
!conv1d_46/StatefulPartitionedCallStatefulPartitionedCall*conv1d_45/StatefulPartitionedCall:output:0conv1d_46_866972conv1d_46_866974*
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
E__inference_conv1d_46_layer_call_and_return_conditional_losses_8667522#
!conv1d_46/StatefulPartitionedCall?
!conv1d_47/StatefulPartitionedCallStatefulPartitionedCall*conv1d_46/StatefulPartitionedCall:output:0conv1d_47_866977conv1d_47_866979*
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
E__inference_conv1d_47_layer_call_and_return_conditional_losses_8667862#
!conv1d_47/StatefulPartitionedCall?
!conv1d_48/StatefulPartitionedCallStatefulPartitionedCall*conv1d_47/StatefulPartitionedCall:output:0conv1d_48_866982conv1d_48_866984*
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
E__inference_conv1d_48_layer_call_and_return_conditional_losses_8668202#
!conv1d_48/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCallcatdense_18_866987dense_18_866989*
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
GPU 2J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_8668472"
 dense_18/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall*conv1d_48/StatefulPartitionedCall:output:0*
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
E__inference_flatten_6_layer_call_and_return_conditional_losses_8668692
flatten_6/PartitionedCall?
concatenate_6/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0"flatten_6/PartitionedCall:output:0*
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
I__inference_concatenate_6_layer_call_and_return_conditional_losses_8668842
concatenate_6/PartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0dense_19_866994dense_19_866996*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_8669042"
 dense_19/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_866999dense_20_867001*
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
GPU 2J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_8669312"
 dense_20/StatefulPartitionedCall?
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0"^conv1d_42/StatefulPartitionedCall"^conv1d_43/StatefulPartitionedCall"^conv1d_44/StatefulPartitionedCall"^conv1d_45/StatefulPartitionedCall"^conv1d_46/StatefulPartitionedCall"^conv1d_47/StatefulPartitionedCall"^conv1d_48/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes|
z:?????????:?????????::::::::::::::::::::2F
!conv1d_42/StatefulPartitionedCall!conv1d_42/StatefulPartitionedCall2F
!conv1d_43/StatefulPartitionedCall!conv1d_43/StatefulPartitionedCall2F
!conv1d_44/StatefulPartitionedCall!conv1d_44/StatefulPartitionedCall2F
!conv1d_45/StatefulPartitionedCall!conv1d_45/StatefulPartitionedCall2F
!conv1d_46/StatefulPartitionedCall!conv1d_46/StatefulPartitionedCall2F
!conv1d_47/StatefulPartitionedCall!conv1d_47/StatefulPartitionedCall2F
!conv1d_48/StatefulPartitionedCall!conv1d_48/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:Q M
+
_output_shapes
:?????????

_user_specified_nameconv:LH
'
_output_shapes
:?????????

_user_specified_namecat
?
a
E__inference_flatten_6_layer_call_and_return_conditional_losses_867831

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
E__inference_conv1d_45_layer_call_and_return_conditional_losses_867715

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
dense_200
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
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
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_networkс{"class_name": "Functional", "name": "model_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv"}, "name": "conv", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_42", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_42", "inbound_nodes": [[["conv", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_43", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_43", "inbound_nodes": [[["conv1d_42", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_44", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_44", "inbound_nodes": [[["conv1d_43", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_45", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_45", "inbound_nodes": [[["conv1d_44", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_46", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_46", "inbound_nodes": [[["conv1d_45", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_47", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_47", "inbound_nodes": [[["conv1d_46", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cat"}, "name": "cat", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_48", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_48", "inbound_nodes": [[["conv1d_47", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["cat", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["conv1d_48", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_6", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_6", "inbound_nodes": [[["dense_18", 0, 0, {}], ["flatten_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["concatenate_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 8, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}], "input_layers": [["conv", 0, 0], ["cat", 0, 0]], "output_layers": [["dense_20", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 5]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 26, 1]}, {"class_name": "TensorShape", "items": [null, 5]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv"}, "name": "conv", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_42", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_42", "inbound_nodes": [[["conv", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_43", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_43", "inbound_nodes": [[["conv1d_42", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_44", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_44", "inbound_nodes": [[["conv1d_43", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_45", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_45", "inbound_nodes": [[["conv1d_44", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_46", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_46", "inbound_nodes": [[["conv1d_45", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_47", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_47", "inbound_nodes": [[["conv1d_46", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cat"}, "name": "cat", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_48", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_48", "inbound_nodes": [[["conv1d_47", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["cat", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["conv1d_48", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_6", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_6", "inbound_nodes": [[["dense_18", 0, 0, {}], ["flatten_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["concatenate_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 8, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}], "input_layers": [["conv", 0, 0], ["cat", 0, 0]], "output_layers": [["dense_20", 0, 0]]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv"}}
?


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv1D", "name": "conv1d_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_42", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 1]}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_43", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_44", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_45", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_46", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_46", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?	

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_47", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_47", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cat", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cat"}}
?	

9kernel
:bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_48", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 64]}}
?

?kernel
@bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}
?
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_6", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 8]}, {"class_name": "TensorShape", "items": [null, 1664]}]}
?

Mkernel
Nbias
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1672}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1672]}}
?

Skernel
Tbias
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 8, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?
Yiter

Zbeta_1

[beta_2
	\decay
]learning_ratem?m?m?m?!m?"m?'m?(m?-m?.m?3m?4m?9m?:m??m?@m?Mm?Nm?Sm?Tm?v?v?v?v?!v?"v?'v?(v?-v?.v?3v?4v?9v?:v??v?@v?Mv?Nv?Sv?Tv?"
	optimizer
 "
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
?
regularization_losses
^metrics
_layer_metrics

`layers
anon_trainable_variables
blayer_regularization_losses
trainable_variables
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
&:$@2conv1d_42/kernel
:@2conv1d_42/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
cmetrics
dlayer_metrics

elayers
fnon_trainable_variables
glayer_regularization_losses
trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_43/kernel
:@2conv1d_43/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
hmetrics
ilayer_metrics

jlayers
knon_trainable_variables
llayer_regularization_losses
trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_44/kernel
:@2conv1d_44/bias
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
#regularization_losses
mmetrics
nlayer_metrics

olayers
pnon_trainable_variables
qlayer_regularization_losses
$trainable_variables
%	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_45/kernel
:@2conv1d_45/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
)regularization_losses
rmetrics
slayer_metrics

tlayers
unon_trainable_variables
vlayer_regularization_losses
*trainable_variables
+	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_46/kernel
:@2conv1d_46/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
/regularization_losses
wmetrics
xlayer_metrics

ylayers
znon_trainable_variables
{layer_regularization_losses
0trainable_variables
1	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_47/kernel
:@2conv1d_47/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
5regularization_losses
|metrics
}layer_metrics

~layers
non_trainable_variables
 ?layer_regularization_losses
6trainable_variables
7	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv1d_48/kernel
:@2conv1d_48/bias
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
;regularization_losses
?metrics
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
<trainable_variables
=	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_18/kernel
:2dense_18/bias
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
Aregularization_losses
?metrics
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
Btrainable_variables
C	variables
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
Eregularization_losses
?metrics
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
Ftrainable_variables
G	variables
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
Iregularization_losses
?metrics
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
Jtrainable_variables
K	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_19/kernel
:2dense_19/bias
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
?
Oregularization_losses
?metrics
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
Ptrainable_variables
Q	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_20/kernel
:2dense_20/bias
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
?
Uregularization_losses
?metrics
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
Vtrainable_variables
W	variables
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
+:)@2Adam/conv1d_42/kernel/m
!:@2Adam/conv1d_42/bias/m
+:)@@2Adam/conv1d_43/kernel/m
!:@2Adam/conv1d_43/bias/m
+:)@@2Adam/conv1d_44/kernel/m
!:@2Adam/conv1d_44/bias/m
+:)@@2Adam/conv1d_45/kernel/m
!:@2Adam/conv1d_45/bias/m
+:)@@2Adam/conv1d_46/kernel/m
!:@2Adam/conv1d_46/bias/m
+:)@@2Adam/conv1d_47/kernel/m
!:@2Adam/conv1d_47/bias/m
+:)@@2Adam/conv1d_48/kernel/m
!:@2Adam/conv1d_48/bias/m
&:$2Adam/dense_18/kernel/m
 :2Adam/dense_18/bias/m
':%	?2Adam/dense_19/kernel/m
 :2Adam/dense_19/bias/m
&:$2Adam/dense_20/kernel/m
 :2Adam/dense_20/bias/m
+:)@2Adam/conv1d_42/kernel/v
!:@2Adam/conv1d_42/bias/v
+:)@@2Adam/conv1d_43/kernel/v
!:@2Adam/conv1d_43/bias/v
+:)@@2Adam/conv1d_44/kernel/v
!:@2Adam/conv1d_44/bias/v
+:)@@2Adam/conv1d_45/kernel/v
!:@2Adam/conv1d_45/bias/v
+:)@@2Adam/conv1d_46/kernel/v
!:@2Adam/conv1d_46/bias/v
+:)@@2Adam/conv1d_47/kernel/v
!:@2Adam/conv1d_47/bias/v
+:)@@2Adam/conv1d_48/kernel/v
!:@2Adam/conv1d_48/bias/v
&:$2Adam/dense_18/kernel/v
 :2Adam/dense_18/bias/v
':%	?2Adam/dense_19/kernel/v
 :2Adam/dense_19/bias/v
&:$2Adam/dense_20/kernel/v
 :2Adam/dense_20/bias/v
?2?
!__inference__wrapped_model_866593?
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
?2?
(__inference_model_6_layer_call_fn_867212
(__inference_model_6_layer_call_fn_867570
(__inference_model_6_layer_call_fn_867109
(__inference_model_6_layer_call_fn_867616?
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
?2?
C__inference_model_6_layer_call_and_return_conditional_losses_867005
C__inference_model_6_layer_call_and_return_conditional_losses_867396
C__inference_model_6_layer_call_and_return_conditional_losses_867524
C__inference_model_6_layer_call_and_return_conditional_losses_866948?
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
*__inference_conv1d_42_layer_call_fn_867643?
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
E__inference_conv1d_42_layer_call_and_return_conditional_losses_867634?
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
*__inference_conv1d_43_layer_call_fn_867670?
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
E__inference_conv1d_43_layer_call_and_return_conditional_losses_867661?
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
*__inference_conv1d_44_layer_call_fn_867697?
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
E__inference_conv1d_44_layer_call_and_return_conditional_losses_867688?
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
*__inference_conv1d_45_layer_call_fn_867724?
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
E__inference_conv1d_45_layer_call_and_return_conditional_losses_867715?
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
*__inference_conv1d_46_layer_call_fn_867751?
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
E__inference_conv1d_46_layer_call_and_return_conditional_losses_867742?
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
*__inference_conv1d_47_layer_call_fn_867778?
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
E__inference_conv1d_47_layer_call_and_return_conditional_losses_867769?
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
*__inference_conv1d_48_layer_call_fn_867805?
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
E__inference_conv1d_48_layer_call_and_return_conditional_losses_867796?
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
)__inference_dense_18_layer_call_fn_867825?
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
D__inference_dense_18_layer_call_and_return_conditional_losses_867816?
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
*__inference_flatten_6_layer_call_fn_867836?
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
E__inference_flatten_6_layer_call_and_return_conditional_losses_867831?
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
.__inference_concatenate_6_layer_call_fn_867849?
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
I__inference_concatenate_6_layer_call_and_return_conditional_losses_867843?
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
)__inference_dense_19_layer_call_fn_867869?
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
D__inference_dense_19_layer_call_and_return_conditional_losses_867860?
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
)__inference_dense_20_layer_call_fn_867889?
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
D__inference_dense_20_layer_call_and_return_conditional_losses_867880?
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
$__inference_signature_wrapper_867268catconv"?
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
!__inference__wrapped_model_866593?!"'(-.349:?@MNSTU?R
K?H
F?C
"?
conv?????????
?
cat?????????
? "3?0
.
dense_20"?
dense_20??????????
I__inference_concatenate_6_layer_call_and_return_conditional_losses_867843?[?X
Q?N
L?I
"?
inputs/0?????????
#? 
inputs/1??????????
? "&?#
?
0??????????
? ?
.__inference_concatenate_6_layer_call_fn_867849x[?X
Q?N
L?I
"?
inputs/0?????????
#? 
inputs/1??????????
? "????????????
E__inference_conv1d_42_layer_call_and_return_conditional_losses_867634d3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????@
? ?
*__inference_conv1d_42_layer_call_fn_867643W3?0
)?&
$?!
inputs?????????
? "??????????@?
E__inference_conv1d_43_layer_call_and_return_conditional_losses_867661d3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_43_layer_call_fn_867670W3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_44_layer_call_and_return_conditional_losses_867688d!"3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_44_layer_call_fn_867697W!"3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_45_layer_call_and_return_conditional_losses_867715d'(3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_45_layer_call_fn_867724W'(3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_46_layer_call_and_return_conditional_losses_867742d-.3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_46_layer_call_fn_867751W-.3?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_47_layer_call_and_return_conditional_losses_867769d343?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_47_layer_call_fn_867778W343?0
)?&
$?!
inputs?????????@
? "??????????@?
E__inference_conv1d_48_layer_call_and_return_conditional_losses_867796d9:3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
*__inference_conv1d_48_layer_call_fn_867805W9:3?0
)?&
$?!
inputs?????????@
? "??????????@?
D__inference_dense_18_layer_call_and_return_conditional_losses_867816\?@/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_18_layer_call_fn_867825O?@/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_dense_19_layer_call_and_return_conditional_losses_867860]MN0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_19_layer_call_fn_867869PMN0?-
&?#
!?
inputs??????????
? "???????????
D__inference_dense_20_layer_call_and_return_conditional_losses_867880\ST/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_20_layer_call_fn_867889OST/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_6_layer_call_and_return_conditional_losses_867831]3?0
)?&
$?!
inputs?????????@
? "&?#
?
0??????????
? ~
*__inference_flatten_6_layer_call_fn_867836P3?0
)?&
$?!
inputs?????????@
? "????????????
C__inference_model_6_layer_call_and_return_conditional_losses_866948?!"'(-.349:?@MNST]?Z
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
0?????????
? ?
C__inference_model_6_layer_call_and_return_conditional_losses_867005?!"'(-.349:?@MNST]?Z
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
0?????????
? ?
C__inference_model_6_layer_call_and_return_conditional_losses_867396?!"'(-.349:?@MNSTf?c
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
0?????????
? ?
C__inference_model_6_layer_call_and_return_conditional_losses_867524?!"'(-.349:?@MNSTf?c
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
0?????????
? ?
(__inference_model_6_layer_call_fn_867109?!"'(-.349:?@MNST]?Z
S?P
F?C
"?
conv?????????
?
cat?????????
p

 
? "???????????
(__inference_model_6_layer_call_fn_867212?!"'(-.349:?@MNST]?Z
S?P
F?C
"?
conv?????????
?
cat?????????
p 

 
? "???????????
(__inference_model_6_layer_call_fn_867570?!"'(-.349:?@MNSTf?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
(__inference_model_6_layer_call_fn_867616?!"'(-.349:?@MNSTf?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
$__inference_signature_wrapper_867268?!"'(-.349:?@MNST_?\
? 
U?R
$
cat?
cat?????????
*
conv"?
conv?????????"3?0
.
dense_20"?
dense_20?????????