# Ordered FP32 arithmetic on the shared sequence model. No separate container.
import Base
import ./math_sequence.bend as Sequence

def multiply_add(accumulator: F32,input: F32,weight: F32) -> F32:
  (accumulator + input * weight : F32)

def weighted_add(weight: F32,accumulator: F32,input: F32) -> F32:
  multiply_add(accumulator,input,weight)

def zip_mac(length: Nat,accumulator: Sequence.Values(F32,length),input: Sequence.Values(F32,length),weight: F32) -> Sequence.Values(F32,length):
  Sequence.zip_map(~F32,~F32,~weighted_add,length,weight,accumulator,input)

# The Cephes expf polynomial after its constant and linear terms.
def exponential_polynomial(+r: F32) -> F32:
  (((((0.00019875691500 * r + 0.0013981999507 : F32) * r + 0.0083334519073 : F32) * r + 0.041665795894 : F32) * r + 0.16666665459 : F32) * r + 0.50000001201 : F32)

# The 32-bit word of a U32, read as an F32, and back.
def from_bits(value: U32) -> F32:
  U32{word} = value
  F32{word}

def to_bits(value: F32) -> U32:
  F32{bits} = value
  U32{bits}

# exp(r)-1-r near |r| <= ln(2)/2. Chebyshev coefficients, rounded to F32;
# paired evaluation shortens the dependency chain without contracting products.
def exponential_remainder(+r: F32) -> F32:
  +z = (r * r : F32)
  (z * ((0.5 + 0.16666577756404877 * r : F32) + z * ((0.041666556149721146 + 0.008363173343241215 * r : F32) + z * 0.0013926175888627768 : F32) : F32) : F32)

# b=RN(1+q). Recover the small term lost in that sum before adding the
# reduced exponential. Writing (1-b)+q keeps the eight lanes together under
# the default vectorizer; q-(b-1) makes it mix unrelated addition trees.
def logistic_denominator(q: F32,+base: F32,r: F32,correction: F32) -> F32:
  (base + (r + (correction + ((1.0 - base : F32) + q : F32) : F32) : F32) : F32)

def selected(+mask: U32,yes: F32,+no: F32) -> F32:
  from_bits(U32.xor(F32.bits(no),U32.and(U32.xor(F32.bits(yes),F32.bits(no)),mask)))

# Above 18 SiLU rounds to x. Pass x through, including positive infinity;
# the negative tail uses the clamped input before the final underflow.
def silu_finish(+x: F32,value: F32) -> F32:
  +word = F32.bits(x)
  +positive = U32.not((0 - U32.shrn(word,31n) : U32))
  +large = U32.and(positive,(0 - U32.shrn((1099956224 - word : U32),31n) : U32))
  selected(large,x,value)

def logistic_scaled(silu: Bool,x: F32,value: F32,power: F32,denominator: F32) -> F32:
  match silu:
    case True{}:
      silu_finish(x,(((value * power : F32) / denominator : F32) * 0.00000000000000000005421010862427522 : F32))
    case False{}:
      ((power / denominator : F32) * 0.00000000000000000005421010862427522 : F32)

# u=-value, n approximates u/ln(2) to the nearest integer, r=u-n*ln(2).
# The exponent word makes
# H=2^(64-n), so sigmoid(x)=H/(exp(r)+2^-n)*2^-64. H remains normal;
# only the last multiplication underflows. The tiny denominator addend q
# may be floored to the least normal F32 without changing its rounded sum.
def sigmoid_reduced(silu: Bool,x: F32,+value: F32,+biased: F32) -> F32:
  +n = (biased - 12582912.0 : F32)
  +r = ((n * F32.neg(0.693359375) - value : F32) + n * 0.00021219444005469058 : F32)
  +power = from_bits((1602224128 - U32.shln(F32.bits(biased),23n) : U32))
  +q = from_bits((U32.max(F32.bits(power),545259520) - 536870912 : U32))
  logistic_scaled(silu,x,value,power,logistic_denominator(q,(1.0 + q : F32),r,exponential_remainder(r)))

def sigmoid_input(silu: Bool,x: F32,+value: F32) -> F32:
  sigmoid_reduced(silu,x,value,(value * F32.neg(1.44269504088896341) + 12582912.0 : F32))

# At x>=18 sigmoid rounds to 1. Below -110 both sigmoid and SiLU round
# to zero; multiplying the clamped x before the final scaling preserves -0
# for SiLU, including at -infinity. NaNs pass the clamp and propagate.
def sigmoid(+x: F32) -> F32:
  sigmoid_input(False{},x,F32.max(F32.min(18.0,x),F32.neg(110.0)))

def activate(+x: F32,enabled: Bool) -> F32:
  match enabled:
    case True{}:
      sigmoid_input(True{},x,F32.max(F32.min(18.0,x),F32.neg(110.0)))
    case False{}:
      x

# Twelve-bit heads have exact normal head products; the residual carries
# the remaining twelve fraction bits.
def split_high(+a: F32) -> F32:
  from_bits(U32.and(F32.bits(a),4294963200))

# Chebyshev interpolation of (exp(r)-1-r)/r^2 on [-0.375,1.0], rounded to
# F32. The compensated head needs a wider interval than ln(2)/2. Paired
# evaluation shortens the dependency chain without contracting products.
def gelu_exponential_remainder(+r: F32) -> F32:
  +z = (r * r : F32)
  (z * ((0.5 + 0.16666662693023682 * r : F32) + z * ((0.04166663810610771 + 0.008334293961524963 * r : F32) + z * ((0.0013882070779800415 + 0.00019411252287682146 * r : F32) + z * 3.194776218151674e-05 : F32) : F32) : F32) : F32)

def gelu_reduced(x: F32,+value: F32,low: F32,+biased: F32) -> F32:
  +n = (biased - 12582912.0 : F32)
  +r = (((n * F32.neg(0.693359375) - value : F32) + n * 0.00021219444005469058 : F32) - low : F32)
  +power = from_bits((1602224128 - U32.shln(F32.bits(biased),23n) : U32))
  +q = from_bits((U32.max(F32.bits(power),545259520) - 536870912 : U32))
  (((x * power : F32) / logistic_denominator(q,(1.0 + q : F32),r,gelu_exponential_remainder(r)) : F32) * 0.00000000000000000005421010862427522 : F32)

# w=2 sqrt(2/pi) c (1+0.044715 c^2). Twelve-bit input/product heads,
# a nine-bit cube head and short coefficient heads keep the head sum exact
# for 0.5<=|c|<=16: the linear/cubic products need at most 18/16 bits,
# and their exponent gap is at most 7/4 bits. Smaller inputs have a small
# absolute head error, measured separately from the ordered-map theorem.
# Carry coefficient and cubic tails through c_low^2, omitting c_low^3.
type Argument is Data:
  Argument{x: F32,head: F32,low: F32}

def gelu_argument(+x: F32,+c: F32) -> Argument:
  +h = split_high(c)
  +l = (c - h : F32)
  +p = (h * h : F32)
  +ph = split_high(p)
  +cube = (h * ph : F32)
  +cubeh = from_bits(U32.and(F32.bits(cube),4294934528))
  +linear = (1.59375 * h : F32)
  +cubic = (0.0712890625 * cubeh : F32)
  +w = (linear + cubic : F32)
  low = (((0.0020191215444356203 * h : F32) +
      ((0.0712890625 * (cube - cubeh : F32) : F32) + (6.575377483386546e-05 * cube : F32) : F32) : F32) +
      (((0.07135481387376785 * h : F32) * (p - ph : F32) : F32) +
        (l * (1.5957691669464111 + 0.21406444907188416 * (p + h * l : F32) : F32) : F32) : F32) : F32)
  +value = F32.max(w,F32.neg(110.0))
  Argument{x,value,low}

# Positive results already round to x at magnitude >=6. Cap their
# negative-tail evaluation there to avoid an unnecessary final underflow;
# negative results keep the full range through magnitude 16.
def gelu_negative_input(+x: F32) -> F32:
  +word = F32.bits(x)
  +negative = (0 - U32.shrn(word,31n) : U32)
  F32.max(from_bits(U32.or(word,2147483648)),selected(negative,F32.neg(16.0),F32.neg(6.0)))

def gelu_finish(arg: Argument) -> F32:
  Argument{+x,+value,low} = arg
  +negative_input = gelu_negative_input(x)
  +tail = gelu_reduced(negative_input,value,low,(value * F32.neg(1.44269504088896341) + 12582912.0 : F32))
  selected((0 - U32.shrn(F32.bits(x),31n) : U32),tail,(x + tail : F32))

# GELU(x)=x+GELU(-x). Evaluate a negative tail on both sides, then
# restore the positive result. Clamping before arithmetic keeps infinities
# out of the products. Negative magnitudes >=16 round to -0.
# The original word selects the sign, including -0; NaNs propagate.
def gelu_prepared(+x: F32) -> Argument:
  gelu_argument(x,gelu_negative_input(x))

def gelu_tanh(x: F32) -> F32:
  gelu_finish(gelu_prepared(x))

# Sign restoration and the tiny interval are word selections. For |x|<2^-12,
# |x-tanh(x)|<|x|^3/3 is below half an F32 ulp, so return x, including -0.
def tanh_finish(+x: F32,+value: F32) -> F32:
  +word = F32.bits(x)
  +small = (0 - U32.shrn((U32.and(word,2147483647) - 964689920 : U32),31n) : U32)
  +keep = U32.or(small,2147483648)
  from_bits(U32.or(U32.and(word,keep),U32.and(F32.bits(value),U32.not(keep))))

# At |x|>=10, 1-tanh(|x|)<2*exp(-20)<2^-25. Clamping before arithmetic
# also keeps infinities away from intermediate products; NaNs propagate.
def tanh_magnitude(+x: F32) -> F32:
  F32.max(F32.min(10.0,from_bits(U32.and(F32.bits(x),2147483647))),0.000244140625)

# Chebyshev interpolation of (tanh(sqrt(z))/sqrt(z)-1)/z, |t|<=0.75.
# The coefficients are F32 values.
def tanh_small(+t: F32,+z: F32) -> F32:
  (t + (t * z : F32) * (((((0.001909503829665482 * z + F32.neg(0.007933800108730793) : F32) * z + 0.021615320816636086 : F32) * z + F32.neg(0.05393586680293083) : F32) * z + 0.1333317905664444 : F32) * z + F32.neg(0.3333333134651184) : F32) : F32)

# One additional term for the expm1 correction in the tanh quotient.
def tanh_remainder(+r: F32) -> F32:
  ((((((0.00019890980911441147 * r + 0.0013933641603216529 : F32) * r + 0.00833331048488617 : F32) * r + 0.04166646674275398 : F32) * r + 0.1666666716337204 : F32) * r + 0.5 : F32) * (r * r : F32) : F32)

def tanh_exponential(+a: F32,+biased: F32,x: F32) -> F32:
  +n = (biased - 12582912.0 : F32)
  +r = (((a * 2.0 : F32) - n * 0.693359375 : F32) - n * F32.neg(0.00021219444005469058) : F32)
  +q = from_bits((1065353216 - U32.shln(F32.bits(biased),23n) : U32))
  +t = F32.min(0.75,a)
  +small = tanh_small(t,(t * t : F32))
  +large = (1.0 - 2.0 * (q / ((1.0 + q : F32) + (r + tanh_remainder(r) : F32) : F32) : F32) : F32)
  +mask = (0 - U32.shrn((F32.bits(a) - 1061158912 : U32),31n) : U32)
  tanh_finish(x,selected(mask,small,large))

def tanh_input(+a: F32,x: F32) -> F32:
  tanh_exponential(a,((a * 2.0 : F32) * 1.44269504088896341 + 12582912.0 : F32),x)

def tanh(+x: F32) -> F32:
  tanh_input(tanh_magnitude(x),x)

# Split 2^n at +64 for nonnegative inputs and -64 for negative inputs.
# The first product is always normal and exact; only the last product can
# underflow. The exponent words come directly from the biased integer n.
def exponential_scale(biased: F32,x: F32,p: F32) -> F32:
  +sign = U32.shrn(U32.and(F32.bits(x),2147483648),1n)
  ((p * from_bits((1602224128 - sign : U32)) : F32) *
    from_bits((528482304 + U32.shln(F32.bits(biased),23n) + sign : U32)) : F32)

# FastTwoSum retains the low bits of 1+r before adding the quadratic tail.
def exponential_precise(+r: F32) -> F32:
  +sum = (1.0 + r : F32)
  +error = (r - (sum - 1.0 : F32) : F32)
  (sum + ((exponential_polynomial(r) * (r * r : F32) : F32) + error : F32) : F32)

def exponential_reduced(+x: F32,+biased: F32) -> F32:
  +n = (biased - 12582912.0 : F32)
  +high = (x - n * 0.693359375 : F32)
  +low = (n * F32.neg(0.00021219444005469058) : F32)
  +r = (high - low : F32)
  exponential_scale(biased,x,exponential_precise(r))

def exponential_input(+x: F32) -> F32:
  exponential_reduced(x,(x * 1.44269504088896341 + 12582912.0 : F32))

def exp(x: F32) -> F32:
  exponential_input(F32.max(F32.min(89.0,x),F32.neg(104.0)))

# For s near r/(2+r), d = r-(2+r)*s gives the exact real identity
# log(1+r) = 2*atanh(s) + log(1+d/(1+s)). The odd series ends at s^11.
# Split r and s into 12-bit heads so their head product is exact; retain
# the low products in d. (1-s)*(1+s^2+s^4) approximates 1/(1+s).
# FastTwoSum retains the low bits when adding the exponent's ln(2) part.
def logarithm_series(+r: F32,+n: F32) -> F32:
  +s = (r / (2.0 + r : F32) : F32)
  +square = (s * s : F32)
  +r_high = from_bits(U32.and(F32.bits(r),4294963200))
  +r_low = (r - r_high : F32)
  +s_high = from_bits(U32.and(F32.bits(s),4294963200))
  +s_low = (s - s_high : F32)
  +product_low = ((r_high * s_low + r_low * s_high : F32) + r_low * s_low : F32)
  +correction = (((r - 2.0 * s : F32) - r_high * s_high : F32) - product_low : F32)
  +p = ((((0.09090909090909091 * square + 0.1111111111111111 : F32) * square + 0.14285714285714285 : F32) * square + 0.2 : F32) * square + 0.3333333333333333 : F32)
  +main = (2.0 * s : F32)
  +rest = ((main * square : F32) * p + correction * ((1.0 - s : F32) * (1.0 + square * (1.0 + square : F32) : F32) : F32) : F32)
  +high = (n * 0.693359375 : F32)
  +sum = (high + main : F32)
  +tail = (main - (sum - high : F32) : F32)
  (sum + (rest + (n * F32.neg(0.000212194440) + tail : F32) : F32) : F32)

# The exponent carry at mantissa word 0x3504f4 selects the same reduced
# interval as halving m above sqrt(2), without a floating-point selection.
def logarithm_normalized(x: F32,adjustment: F32) -> F32:
  +word = F32.bits(x)
  +adjusted = (word + 4913932 : U32)
  logarithm_series((from_bits((word - U32.and(adjusted,4286578688) + 1065353216 : U32)) - 1.0 : F32),
    ((from_bits(U32.or(U32.and(U32.shrn(adjusted,23n),255),1258291200)) - 8388735.0 : F32) - adjustment : F32))

# Scaling a subnormal by 2^23 is exact. Zero, negatives, infinities and
# NaNs are selected explicitly, after normalizing the finite positive input.
def logarithm_finish(+x: F32,value: F32) -> F32:
  +word = F32.bits(x)
  +magnitude = U32.and(word,2147483647)
  +zero = (0 - U32.shrn((magnitude - 1 : U32),31n) : U32)
  +negative = (0 - U32.shrn(word,31n) : U32)
  +special = U32.not((0 - U32.shrn((magnitude - 2139095040 : U32),31n) : U32))
  selected(zero,from_bits(4286578688),selected(negative,from_bits(2143289344),selected(special,(x + x : F32),value)))

def logarithm_input(+x: F32) -> F32:
  +small = (0 - U32.shrn((U32.and(F32.bits(x),2147483647) - 8388608 : U32),31n) : U32)
  logarithm_finish(x,logarithm_normalized(selected(small,(x * 8388608.0 : F32),x),from_bits(U32.and(small,1102577664))))

def log(x: F32) -> F32:
  logarithm_input(x)

# The square root rounded to nearest, as IEEE 754 requires of every backend.
def sqrt(x: F32) -> F32:
  F32.sqrt(x)

# Cancellation-free even derivatives with one exponential reduction.
def sigmoid_derivative(x: F32) -> F32:
  +q = exp(F32.neg(from_bits(U32.and(F32.bits(x),2147483647))))
  (q / (1.0 + q * (2.0 + q : F32) : F32) : F32)

def tanh_derivative(x: F32) -> F32:
  +q = exp((F32.neg(2.0) * from_bits(U32.and(F32.bits(x),2147483647)) : F32))
  ((4.0 * q : F32) / (1.0 + q * (2.0 + q : F32) : F32) : F32)

# The derivative divided by the distance from its zero, a degree-eight
# polynomial on [-2,-0.5] in the exact difference from the head of the zero.
def silu_derivative_root(+value: F32) -> F32:
  +d = (value + 1.2784645557403564 : F32)
  +z = (d * d : F32)
  +p0 = (0.2178117036819458 + 0.14664886891841888 * d : F32)
  +p1 = (0.01887473464012146 + F32.neg(0.01522465143352747) * d : F32)
  +p2 = (F32.neg(0.006606101058423519) + 0.0001401008339598775 * d : F32)
  +p3 = (0.0007981295930221677 + 0.00015176329179666936 * d : F32)
  +tail = (p0 + z * (p1 + z * (p2 + z * (p3 + z * F32.neg(4.2798692447831854e-05) : F32) : F32) : F32) : F32)
  ((d - 1.2979282537628478e-08 : F32) * tail : F32)

def silu_derivative_tail(+value: F32,+scaled: F32) -> F32:
  +s = (scaled * 0.00000000000000000005421010862427522 : F32)
  +word = F32.bits(value)
  +above = U32.not((0 - U32.shrn((word - 3204448256 : U32),31n) : U32))
  +below = U32.not((0 - U32.shrn((3221225472 - word : U32),31n) : U32))
  +tail = ((scaled * (1.0 + (value - value * s : F32) : F32) : F32) * 0.00000000000000000005421010862427522 : F32)
  selected(U32.and(above,below),silu_derivative_root(value),tail)

type DerivativeArgument is Data:
  DerivativeArgument{x: F32,value: F32,scaled: F32}

def silu_derivative_prepared(+x: F32,+value: F32) -> DerivativeArgument:
  +biased = (value * F32.neg(1.44269504088896341) + 12582912.0 : F32)
  +n = (biased - 12582912.0 : F32)
  +r = ((n * F32.neg(0.693359375) - value : F32) + n * 0.00021219444005469058 : F32)
  +power = from_bits((1602224128 - U32.shln(F32.bits(biased),23n) : U32))
  +q = from_bits((U32.max(F32.bits(power),545259520) - 536870912 : U32))
  DerivativeArgument{x,value,(power / logistic_denominator(q,(1.0 + q : F32),r,exponential_remainder(r)) : F32)}

def silu_derivative_prepare(+x: F32) -> DerivativeArgument:
  silu_derivative_prepared(x,F32.max(from_bits(U32.or(F32.bits(x),2147483648)),F32.neg(110.0)))

def silu_derivative_finish(arg: DerivativeArgument) -> F32:
  DerivativeArgument{+x,value,scaled} = arg
  +tail = silu_derivative_tail(value,scaled)
  +magnitude = U32.and(F32.bits(x),2147483647)
  +nan = (0 - U32.shrn((2139095040 - magnitude : U32),31n) : U32)
  selected(nan,x,selected((0 - U32.shrn(F32.bits(x),31n) : U32),tail,(1.0 - tail : F32)))

def silu_derivative(x: F32) -> F32:
  silu_derivative_finish(silu_derivative_prepare(x))

# Root quotient on [-1.25,-0.5]. Split the linear head exactly and
# retain the low parts of the first two rounded quotient coefficients.
def gelu_derivative_root(+c: F32) -> F32:
  +d = (c + 0.7524614334106445 : F32)
  +dh = from_bits(U32.and(F32.bits(d),4294967292))
  +dl = (d - dh : F32)
  +z = (d * d : F32)
  +p0 = (0.05540008842945099 + 0.38751843571662903 * d : F32)
  +p1 = (F32.neg(0.01578284427523613) + F32.neg(0.11394448578357697) * d : F32)
  +p2 = (F32.neg(0.01661972515285015) + 0.01968197152018547 * d : F32)
  +p3 = (0.005269420798867941 + F32.neg(0.0024054001551121473) * d : F32)
  +p4 = (F32.neg(0.0009768777526915073) + 8.785235695540905e-05 * d : F32)
  +tail = (p0 + z * (p1 + z * (p2 + z * (p3 + z * p4 : F32) : F32) : F32) : F32)
  ((0.375 * dh : F32) + (((d - 1.1339628272762046e-08 : F32) * tail : F32) + (((0.375 * dl : F32) - 4.25236024170772e-09 : F32) + d * (F32.neg(1.8141816937600197e-09) + d * 1.0859928956108433e-08 : F32) : F32) : F32) : F32)

type GeluPrepared is Data:
  GeluPrepared{x: F32,head: F32,low: F32,slope: F32,slope_error: F32}

def gelu_derivative_prepared(+c: F32,arg: Argument) -> GeluPrepared:
  Argument{x,head,+low} = arg
  +h = split_high(c)
  +l = (c - h : F32)
  +cube = (h * split_high((h * h : F32)) : F32)
  +cubeh = from_bits(U32.and(F32.bits(cube),4294934528))
  +slope = (1.59375 * h + 3.0 * (0.0712890625 * cubeh : F32) : F32)
  +slope_error = ((3.0 * low : F32) - ((0.004038243088871241 * h : F32) + (3.1915383338928223 * l : F32) : F32) : F32)
  GeluPrepared{x,head,low,slope,slope_error}

def gelu_derivative_prepare(+x: F32) -> GeluPrepared:
  +c = F32.max(from_bits(U32.or(F32.bits(x),2147483648)),F32.neg(11.0))
  gelu_derivative_prepared(c,gelu_argument(x,c))

type GeluReduced is Data:
  GeluReduced{x: F32,power: F32,denominator: F32,error: F32,slope: F32,slope_error: F32}

def gelu_derivative_exponential(arg: GeluPrepared) -> GeluReduced:
  GeluPrepared{x,+head,low,slope,slope_error} = arg
  +biased = (head * F32.neg(1.44269504088896341) + 12582912.0 : F32)
  +n = (biased - 12582912.0 : F32)
  +rh = (n * F32.neg(0.693359375) - head : F32)
  +rl = (n * 0.00021219444005469058 - low : F32)
  +r = (rh + rl : F32)
  +v = (r - rh : F32)
  +rerror = ((rh - (r - v : F32) : F32) + (rl - v : F32) : F32)
  +power = from_bits((1602224128 - U32.shln(F32.bits(biased),23n) : U32))
  +small = from_bits((U32.max(F32.bits(power),545259520) - 536870912 : U32))
  +base = (1.0 + small : F32)
  +tail = (r + (gelu_exponential_remainder(r) + ((1.0 - base : F32) + small : F32) : F32) : F32)
  +denominator = (base + tail : F32)
  +error = (((base - denominator : F32) + tail : F32) + (denominator - small : F32) * rerror : F32)
  GeluReduced{x,power,denominator,error,slope,slope_error}

# Correct the quotient before the final power-of-two scaling.
def gelu_derivative_evaluated(arg: GeluReduced) -> DerivativeArgument:
  GeluReduced{+x,+power,+denominator,error,+slope,+slope_error} = arg
  +sigmoid = ((power / denominator : F32) * 0.00000000000000000005421010862427522 : F32)
  +factor = (1.0 + slope : F32)
  +residual = (slope_error - (slope + slope_error : F32) * sigmoid : F32)
  +quotient = split_high((factor / denominator : F32))
  +dh = split_high(denominator)
  +dl = (denominator - dh : F32)
  +remainder = (((factor - quotient * dh : F32) - quotient * dl : F32) + (residual - quotient * error : F32) : F32)
  +value = (quotient + remainder / denominator : F32)
  +result = ((power * value : F32) * 0.00000000000000000005421010862427522 : F32)
  DerivativeArgument{x,F32.max(from_bits(U32.or(F32.bits(x),2147483648)),F32.neg(11.0)),result}

# Split the linear term before adding the odd correction to one half.
def gelu_derivative_small(+x: F32) -> F32:
  +h = split_high(x)
  +l = (x - h : F32)
  +linear = (0.796875 * h : F32)
  +base = (0.5 + linear : F32)
  +z = (x * x : F32)
  +p = (F32.neg(0.26727786660194397) + z * (0.061208296567201614 + z * (F32.neg(0.009932546876370907) + z * 0.0011246189242228866 : F32) : F32) : F32)
  +low = ((0.5 - base : F32) + linear : F32)
  (base + (low + ((0.0010095608028653559 * h + 0.7978845834732056 * l : F32) + (x * z : F32) * p : F32) : F32) : F32)

def gelu_derivative_finish(arg: DerivativeArgument) -> F32:
  DerivativeArgument{+x,+value,result} = arg
  +word = F32.bits(value)
  +above = U32.not((0 - U32.shrn((word - 3204448256 : U32),31n) : U32))
  +below = U32.not((0 - U32.shrn((3214934016 - word : U32),31n) : U32))
  +tail = selected(U32.and(above,below),gelu_derivative_root(value),result)
  +magnitude = U32.and(F32.bits(x),2147483647)
  +nan = (0 - U32.shrn((2139095040 - magnitude : U32),31n) : U32)
  +small = (0 - U32.shrn((magnitude - 1056964608 : U32),31n) : U32)
  selected(nan,x,selected(small,gelu_derivative_small(x),selected((0 - U32.shrn(F32.bits(x),31n) : U32),tail,(1.0 - tail : F32))))

def gelu_tanh_derivative(x: F32) -> F32:
  gelu_derivative_finish(gelu_derivative_evaluated(gelu_derivative_exponential(gelu_derivative_prepare(x))))
