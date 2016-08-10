#ifndef __DISTRIBUTIONS
#define __DISTRIBUTIONS

template <class TRange>
class UniformRealDistribution {
	private:
		TRange a, b, scale;
	public:
		UniformRealDistribution() {a=0; b=1; scale=1;} 
		template <class Tgenerator>
		UniformRealDistribution(const TRange a, const TRange b,
				Tgenerator generator)
			: a(a), b(b) {
				scale = (b-a) / ((TRange)generator.max() - generator.min() + 1); 
			};

		template <class Tgenerator>
		TRange operator () (Tgenerator &generator) {
			return scale * generator() + a;
		}
};

#endif
