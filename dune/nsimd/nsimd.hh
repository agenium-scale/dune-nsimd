#ifndef DUNE_NSIMD_NSIMD_HH
#define DUNE_NSIMD_NSIMD_HH

#include <cstddef>
#include <ostream>
#include <type_traits>
#include <utility>

#include <dune/common/math.hh>
#include <dune/common/rangeutilities.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/simd/loop.hh>
#include <dune/common/simd/simd.hh>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#include <dune/nsimd/upstream/nsimd.h>
#pragma GCC diagnostic pop

namespace Dune {
  namespace Simd {
    namespace NsimdImpl {

      /* true for all nsimd-types*/
      template<class T>
      struct IsVector : std::false_type {};

      /* true for all boolean nsimd-types*/
      template<class T, typename = void>
      struct IsMask : std::false_type {};

      template<class T>
      struct IsMask<T, std::enable_if_t<IsVector<T>::value && std::is_same<Scalar<T>,bool>::value>>
             : std::true_type {};

      // construct a nsimd vector for a given scalar and size
      // falls back to loopsimd
      template<class S, std::size_t size>
      struct Vector {
        using type = LoopSIMD<S, size>;
      };

      // construct a nsimd mask for a given nsimd vector
      template<class V>
      struct Mask;
    } //namespace NsimdImpl

    // declare a nsimd vector and corresponding mask
    // the mask name is constructed from the vector name by appending a "b"
#define DUNE_NSIMD_NSIMD_SIMD_TYPES(VECTOR, SCALAR, SIZE)   \
    namespace NsimdImpl {                                         \
      template<> struct IsVector<VECTOR>    : std::true_type {};        \
      template<> struct IsVector<VECTOR##b> : std::true_type {};        \
      template<> struct Vector<SCALAR, SIZE> { using type = VECTOR; };  \
      template<> struct Mask<VECTOR>         { using type = VECTOR##b; }; \
    }                                                                   \
    /* Overloads necessary for SIMD-interface-compatibility */          \
    namespace Overloads {                                               \
      template<> struct ScalarType<VECTOR>    { using type = SCALAR; }; \
      template<> struct ScalarType<VECTOR##b> { using type = bool; };   \
      template<> struct LaneCount<VECTOR>    : index_constant<SIZE> {}; \
      template<> struct LaneCount<VECTOR##b> : index_constant<SIZE> {}; \
    }                                                                   \
    static_assert(true, "unfudge automatic indent")

// instruction set SSE2 or higher
#if MAX_VECTOR_SIZE >= 128
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec16c, char,   16);
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec8s,  short,   8);
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec4i,  int,     4);
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec2q,  long,    2);

    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec4f,  float,   4);
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec2d,  double,  2);
#endif

// instruction set AVX or higher (or emulated)
#if MAX_VECTOR_SIZE >= 256
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec8f,  float,   8);
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec4d,  double,  4);

// instruction set AVX2 or higher (or emulated)
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec32c, char,   32);
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec16s, short,  16);
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec8i,  int,     8);
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec4q,  long,    4);
#endif

// instruction set AVX512 or higher (or emulated)
#if MAX_VECTOR_SIZE >= 512
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec16i, int,    16);
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec8q,  long,    8);

    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec16f, float,  16);
    DUNE_NSIMD_NSIMD_SIMD_TYPES(Vec8d,  double,  8);
#endif

    namespace NsimdImpl {

      template<class V>
      class Proxy
      {
        static_assert(std::is_same<V, std::decay_t<V> >::value, "Class Proxy "
                      "may only be instantiated with unqualified types");

      public:
        using value_type = Scalar<V>;

      private:
        static_assert(std::is_arithmetic<value_type>::value,
                      "Only artihmetic types are supported");
        V &vec_;
        std::size_t idx_;

      public:
        Proxy(std::size_t idx, V &vec)
          : vec_(vec), idx_(idx)
        { }

        Proxy(const Proxy&) = delete;
        // allow move construction so we can return proxies from functions in
        // C++14
        Proxy(Proxy&&) = default;

        operator value_type() const { return vec_[idx_]; }
        template<class W, class = std::enable_if_t<IsVector<W>::value> >
        operator W() const { return value_type(*this); }

        // assignment operators
        template<class T,
                 class = decltype(std::declval<value_type&>() =
                                  autoCopy(std::declval<T>()) )>
        Proxy operator=(T &&o) &&
        {
          vec_.insert(idx_, autoCopy(std::forward<T>(o)));
          return { idx_, vec_ };
        }
#define DUNE_SIMD_VCL_ASSIGNMENT(OP)                                    \
        template<class T,                                               \
                 class = decltype(std::declval<value_type&>() OP##=     \
                                  autoCopy(std::declval<T>()) )>        \
        Proxy operator OP##=(T &&o) &&                                  \
        {                                                               \
          vec_.insert(idx_, vec_[idx_] OP autoCopy(std::forward<T>(o))); \
          return { idx_, vec_ };                                        \
        }                                                               \
        template<class T,                                               \
                 class = decltype(std::declval<T&>() OP##=              \
                                  std::declval<value_type>()),          \
                 class = std::enable_if_t<IsVector<T>::value> >         \
        friend decltype(auto) operator OP##=(T &l, Proxy &&r)           \
        {                                                               \
          return l OP##= value_type(std::move(r));                      \
        }
        DUNE_SIMD_VCL_ASSIGNMENT(*);
        DUNE_SIMD_VCL_ASSIGNMENT(/);
        DUNE_SIMD_VCL_ASSIGNMENT(%);
        DUNE_SIMD_VCL_ASSIGNMENT(+);
        DUNE_SIMD_VCL_ASSIGNMENT(-);
        DUNE_SIMD_VCL_ASSIGNMENT(<<);
        DUNE_SIMD_VCL_ASSIGNMENT(>>);
        DUNE_SIMD_VCL_ASSIGNMENT(&);
        DUNE_SIMD_VCL_ASSIGNMENT(^);
        DUNE_SIMD_VCL_ASSIGNMENT(|);
#undef DUNE_SIMD_VCL_ASSIGNMENT

        // unary (prefix) operators
        template<class T = value_type,
                 class = std::enable_if_t<!std::is_same<T, bool>::value> >
        Proxy operator++() { ++(vec_[idx_]); return *this; }
        template<class T = value_type,
                 class = std::enable_if_t<!std::is_same<T, bool>::value> >
        Proxy operator--() { --(vec_[idx_]); return *this; }

        // postfix operators
        template<class T = value_type,
                 class = std::enable_if_t<!std::is_same<T, bool>::value> >
        value_type operator++(int) { return vec_[idx_]++; }
        template<class T = value_type,
                 class = std::enable_if_t<!std::is_same<T, bool>::value> >
        value_type operator--(int) { return vec_[idx_]--; }

        // swap on proxies swaps the proxied vector entries.  As such, it
        // applies to rvalues of proxies too, not just lvalues
        friend void swap(const Proxy &a, const Proxy &b) {
          value_type tmp = a.vec_[a.idx_];
          a.vec_.insert(a.idx_, b.vec_[b.idx_]);
          b.vec_.insert(b.idx_, tmp);
        }
        friend void swap(value_type &a, const Proxy &b) {
          value_type tmp = a;
          a = b.vec_[b.idx_];
          b.vec_.insert(b.idx_, tmp);
        }
        friend void swap(const Proxy &a, value_type &b) {
          value_type tmp = a.vec_[a.idx_];
          a.vec_.insert(a.idx_, b);
          b = tmp;
        }

        // These operators should not be necessary.  However, since there is a
        // conversion operator to vector types (which is needed in order to
        // allow broadcast assignement from proxy to vector), most of these
        // operation result in ambigous overloads, and we provide these
        // operators to resolve the ambiguities.
#define DUNE_NSIMD_BINARY(OP)                                     \
        template<class W>                                               \
        friend auto operator OP(const W &l, Proxy&& r)                  \
          -> decltype(l OP std::declval<value_type&&>())                \
        {                                                               \
          return l OP value_type(std::move(r));                         \
        }                                                               \
        template<class W>                                               \
        auto operator OP(const W &r) &&                                 \
          -> decltype(std::declval<value_type&&>() OP r)                \
        {                                                               \
          return value_type(std::move(*this)) OP r;                     \
        }                                                               \
        template<class W>                                               \
        auto operator OP(Proxy<W> &&r) &&                               \
          -> decltype(std::declval<value_type&&>() OP                   \
                      typename Proxy<W>::value_type(std::move(r)))      \
        {                                                               \
          return value_type(std::move(*this)) OP                        \
            typename Proxy<W>::value_type(std::move(r));                \
        }

        DUNE_NSIMD_BINARY(*);
        DUNE_NSIMD_BINARY(/);
        DUNE_NSIMD_BINARY(%);
        DUNE_NSIMD_BINARY(+);
        DUNE_NSIMD_BINARY(-);
        DUNE_NSIMD_BINARY(<<);
        DUNE_NSIMD_BINARY(>>);
        DUNE_NSIMD_BINARY(&);
        DUNE_NSIMD_BINARY(^);
        DUNE_NSIMD_BINARY(|);
        DUNE_NSIMD_BINARY(<);
        DUNE_NSIMD_BINARY(>);
        DUNE_NSIMD_BINARY(<=);
        DUNE_NSIMD_BINARY(>=);
        DUNE_NSIMD_BINARY(==);
        DUNE_NSIMD_BINARY(!=);
#undef DUNE_NSIMD_BINARY
      };

    } // namespace NsimdImpl

#undef DUNE_NSIMD_NSIMD_SIMD_TYPES

    namespace Overloads {
      template<class S, class V>
      struct RebindType<S, V,
                        std::enable_if_t<!std::is_same<S, bool>::value &&
                                         NsimdImpl::IsVector<V>::value> >
        : NsimdImpl::Vector<S, Simd::lanes<V>()>
      {};
      template<class V>
      struct RebindType<bool, V,
                        std::enable_if_t<NsimdImpl::IsVector<V>::value &&
                                         !NsimdImpl::IsMask<V>::value> >
        : NsimdImpl::Mask<V>
      {};
      template<class V>
      struct RebindType<bool, V,
                        std::enable_if_t<NsimdImpl::IsMask<V>::value> >
      {
        using type = V;
      };

      template<class Vec, typename = std::enable_if_t<NsimdImpl::IsVector<Vec>::value>>
      Scalar<Vec> lane(ADLTag<5>, std::size_t l, const Vec &v) {
        return v.extract(l);
      }

      template<class Vec, typename = std::enable_if_t<NsimdImpl::IsVector<Vec>::value>>
      NsimdImpl::Proxy<Vec> lane(ADLTag<5>, std::size_t l, Vec &v) {
        return {l, v};
      }

      template<class Vec, typename = std::enable_if_t<NsimdImpl::IsVector<Vec>::value &&
                                                      !std::is_reference<Vec>::value>>
      Scalar<Vec> lane(ADLTag<5>, std::size_t l, Vec &&v) {
        return std::forward<Vec>(v)[l];
      }

      template<class Vec, typename = std::enable_if_t<NsimdImpl::IsVector<Vec>::value &&
                                                      !NsimdImpl::IsMask<Vec>::value>>
      auto cond(ADLTag<5>, Mask<Vec> mask, Vec ifTrue, Vec ifFalse) {
        return select(mask, ifTrue, ifFalse);
      }

      template<class Vec, typename = std::enable_if_t<NsimdImpl::IsVector<Vec>::value &&
                                                      NsimdImpl::IsMask<Vec>::value>>
      auto cond(ADLTag<5>, Vec mask, Vec ifTrue, Vec ifFalse) {
        return (mask & ifTrue) | ((!mask) & ifFalse);
      }

      template<class Vec>
      auto max(ADLTag<5, NsimdImpl::IsVector<Vec>::value &&
                         !NsimdImpl::IsMask<Vec>::value>,
               Vec v1, Vec v2)
      {
        return Simd::cond(v1 < v2, v2, v1);
      }

      template<class Vec>
      auto max(ADLTag<5, NsimdImpl::IsMask<Vec>::value>,
               Vec v1, Vec v2)
      {
        return v1 | v2;
      }

      template<class Vec>
      auto min(ADLTag<5, NsimdImpl::IsVector<Vec>::value &&
                         !NsimdImpl::IsMask<Vec>::value>,
               Vec v1, Vec v2)
      {
        return Simd::cond(v1 < v2, v1, v2);
      }

      template<class Vec>
      auto min(ADLTag<5, NsimdImpl::IsMask<Vec>::value>,
               Vec v1, Vec v2)
      {
        return v1 & v2;
      }

      template<class M, typename = std::enable_if_t<NsimdImpl::IsMask<M>::value>>
      bool anyTrue(ADLTag<5>, M mask) {
        return horizontal_or(mask);
      }
      template<class M, typename = std::enable_if_t<NsimdImpl::IsMask<M>::value>>
      bool allTrue(ADLTag<5>, M mask) {
        return horizontal_and(mask);
      }
      template<class M, typename = std::enable_if_t<NsimdImpl::IsMask<M>::value>>
      bool anyFalse(ADLTag<5>, M mask) {
        return !horizontal_and(mask);
      }
      template<class M, typename = std::enable_if_t<NsimdImpl::IsMask<M>::value>>
      bool allFalse(ADLTag<5>, M mask) {
        return !horizontal_or(mask);
      }

      template<class M, typename = std::enable_if_t<NsimdImpl::IsMask<M>::value>>
      auto MaskOr(ADLTag<5>, M &m1, M &m2) {
        return m1 | m2;
      }

      template<class M, typename = std::enable_if_t<NsimdImpl::IsMask<M>::value>>
      auto MaskOr(ADLTag<5>, M &m, bool b) {
        return m | M(b);
      }

      template<class M, typename = std::enable_if_t<NsimdImpl::IsMask<M>::value>>
      auto MaskOr(ADLTag<5>, bool b, M &m) { return MaskOr(m,b); }

      template<class M, typename = std::enable_if_t<NsimdImpl::IsMask<M>::value>>
      auto MaskAnd(ADLTag<5>, M &m1, M &m2) {
        return m1 & m2;
      }

      template<class M, typename = std::enable_if_t<NsimdImpl::IsMask<M>::value>>
      auto MaskAnd(ADLTag<5>, M &m, bool b) {
        return m & M(b);
      }

      template<class M, typename = std::enable_if_t<NsimdImpl::IsMask<M>::value>>
      auto MaskAnd(ADLTag<5>, bool b, M &m) { return MaskAnd(m,b); }

    } //namespace Overloads
  } //namespace Simd

  namespace MathOverloads {
    template<class Vec, typename = std::enable_if_t<Simd::NsimdImpl::IsVector<Vec>::value>>
    auto isNaN(const Vec &v , PriorityTag<3>, ADLTag) {
      Simd::Mask<Vec> out{false};
      for(auto l : range(Simd::lanes(v)))
        Simd::lane(l, out) = Dune::isNaN(Simd::lane(l, v));
      return out;
    }

    template<class Vec, typename = std::enable_if_t<Simd::NsimdImpl::IsVector<Vec>::value>>
    auto isInf(const Vec &v , PriorityTag<3>, ADLTag) {
      Simd::Mask<Vec> out{false};
      for(auto l : range(Simd::lanes(v)))
        Simd::lane(l, out) = Dune::isInf(Simd::lane(l, v));
      return out;
    }

    template<class Vec, typename = std::enable_if_t<Simd::NsimdImpl::IsVector<Vec>::value>>
    auto isFinite(const Vec &v , PriorityTag<3>, ADLTag) {
      Simd::Mask<Vec> out{false};
      for(auto l : range(Simd::lanes(v)))
        Simd::lane(l, out) = Dune::isFinite(Simd::lane(l, v));
      return out;
    }
  } //namespace MathOverloads

  template<class Vec>
  struct AutonomousValueType<Simd::NsimdImpl::Proxy<Vec>> :
    AutonomousValueType<typename Simd::NsimdImpl::Proxy<Vec>::value_type> {};

  namespace Simd {
    namespace NsimdImpl {
      // binary operators
      //
      // Normally, these are provided by the conversion operator in
      // combination with C++'s builtin binary operators.  Other classes
      // that need to provide the binary operators themselves should either
      // 1. deduce the "foreign" operand type independently, i.e. use
      //      template<class... Args, class Foreign>
      //      auto operator@(MyClass<Args...>, Foreign);
      //    or
      // 2. not deduce anything from the foreign argument, i.e.
      //      template<class... Args>
      //      auto operator@(MyClass<Args...>,
      //                     typename MyClass<Args...>::value_type);
      //    or
      //      template<class T, class... Args>
      //      struct MyClass {
      //        auto operator@(T);
      //      }
      //    or
      //      template<class T, class... Args>
      //      struct MyClass {
      //        friend auto operator@(MyClass, T);
      //      }
      //
      // This allows either for an exact match (in the case of option 1.) or
      // for conversions to be applied to the foreign argument (options 2.).
      // In contrast, allowing some of the template parameters being deduced
      // from the self argument also being deduced from the foreign argument
      // will likely lead to ambigous deduction when the foreign argument is
      // a proxy:
      //   template<class T, class... Args>
      //   auto operator@(MyClass<T, Args...>, T);
      // One class that suffers from this problem ist std::complex.
      //
      // Note that option 1. is a bit dangerous, as the foreign argument is
      // catch-all.  This seems tempting in the case of a proxy class, as
      // the operator could just be forwarded to the proxied object with the
      // foreign argument unchanged, immediately creating interoperability
      // with arbitrary foreign classes.  However, if the foreign class also
      // choses option 1., this will result in ambigous overloads, and there
      // is no clear guide to decide which class should provide the overload
      // and which should not.
      //
      // Fortunately, deferring to the conversion and the built-in operators
      // mostly works in the case of this proxy class, because only built-in
      // types can be proxied anyway.  Unfortunately, the Vc vectors and
      // arrays suffer from a slightly different problem.  They chose option
      // 1., but they can't just accept the argument type they are given,
      // since they need to somehow implement the operation in terms of
      // intrinsics.  So they check the argument whether it is one of the
      // expected types, and remove the operator from the overload set if it
      // isn't via SFINAE.  Of course, this proxy class is not one of the
      // expected types, even though it would convert to them...
      //
      // So what we have to do here, unfortunately, is to provide operators
      // for the Vc types explicitly, and hope that there won't be some Vc
      // version that gets the operators right, thus creating ambigous
      // overloads.  Well, if guess it will be #ifdef time if it comes to
      // that.
#define DUNE_NSIMD_BINARY(OP)                             \
      template<class L, class R,                                \
               std::enable_if_t<IsVector<L>::value>* = nullptr> \
      auto operator OP(const L &l, Proxy<R> &&r)                \
        -> decltype(l OP autoCopy(r))                           \
      {                                                         \
        return l OP autoCopy(r);                                \
      }

      DUNE_NSIMD_BINARY(<);
      DUNE_NSIMD_BINARY(>);
      DUNE_NSIMD_BINARY(<=);
      DUNE_NSIMD_BINARY(>=);
      DUNE_NSIMD_BINARY(==);
      DUNE_NSIMD_BINARY(!=);

#undef DUNE_NSIMD_BINARY

    } // namespace NsimdImpl
  } // namespace Simd

} //namespace Dune

#ifdef VCL_NAMESPACE
namespace VCL_NAMESPACE {
#endif

  template<class Vec,
           typename = std::enable_if_t<Dune::Simd::NsimdImpl::IsVector<Vec>::value>>
  std::ostream& operator<<(std::ostream &os, const Vec &v) {
    os << "[";
    for(auto l : Dune::range(Dune::Simd::lanes(v)))
      os << v[l] << ", ";
    os << "]";
    return os;
  }

  //////////////////////////////////////////////////////////////////////
  //
  // Fallback operations
  //

  // The SFINAE prevents ambigous overloads: scalars can be converted
  // implicitly to vectors, so when specifying the RHS formal argument as
  // `Type other` this would be a viable overload for division by scalar.  But
  // an overload for devision by scalar exists already in some cases.  So
  // accept any type as the second argument, but then reject inappropriate
  // types by making this overload non-viable.
#define DUNE_NSIMD_OPASSIGN_V(Type, op)                           \
  template<class Other,                                                 \
           class = std::enable_if_t<Dune::Simd::lanes<Other>() ==       \
                                    Dune::Simd::lanes<Type>()>>         \
  Type &operator op##=(Type &self, Other other)                         \
  {                                                                     \
    for(auto l : Dune::range(Dune::Simd::lanes(self)))                  \
      Dune::Simd::lane(l, self) op##= Dune::Simd::lane(l, other);       \
    return self;                                                        \
  }                                                                     \
  static_assert(true, "Unfudge editor indentation heuristics")

#define DUNE_NSIMD_OPASSIGN_S(Type, op)                   \
  inline Type &                                                 \
  operator op##=(Type &self, Dune::Simd::Scalar<Type> other)    \
  {                                                             \
    for(auto l : Dune::range(Dune::Simd::lanes(self)))          \
      Dune::Simd::lane(l, self) op##= other;                    \
    return self;                                                \
  }                                                             \
  static_assert(true, "Unfudge editor indentation heuristics")

#define DUNE_NSIMD_OPINFIX_SV(Type, op)                   \
  inline Type operator op(Dune::Simd::Scalar<Type> a, Type b)   \
  {                                                             \
    auto tmp = Type{a};                                         \
    return tmp op##= b;                                         \
  }                                                             \
  static_assert(true, "Unfudge editor indentation heuristics")

#define DUNE_NSIMD_OPINFIX_VV(Type, op)                           \
  template<class B,                                                     \
           class = std::enable_if_t<Dune::Simd::lanes<Type>() ==        \
                                    Dune::Simd::lanes<B>()> >           \
  Type operator op(Type a, B b) { return a op##= b; }                   \
  static_assert(true, "Unfudge editor indentation heuristics")

#define DUNE_NSIMD_OPINFIX_VS(Type, op)                   \
  inline Type operator op(Type a, Dune::Simd::Scalar<Type> b)   \
  {                                                             \
    return a op##= b;                                           \
  }                                                             \
  static_assert(true, "Unfudge editor indentation heuristics")

#if MAX_VECTOR_SIZE >= 128
  // Vec4i
  DUNE_NSIMD_OPASSIGN_V(Vec4i, /);
  DUNE_NSIMD_OPINFIX_SV(Vec4i, /);
  DUNE_NSIMD_OPINFIX_VV(Vec4i, /);

  DUNE_NSIMD_OPASSIGN_V(Vec4i, %);
  DUNE_NSIMD_OPASSIGN_S(Vec4i, %);
  DUNE_NSIMD_OPINFIX_SV(Vec4i, %);
  DUNE_NSIMD_OPINFIX_VV(Vec4i, %);
  DUNE_NSIMD_OPINFIX_VS(Vec4i, %);

  DUNE_NSIMD_OPASSIGN_V(Vec4i, <<);
  DUNE_NSIMD_OPINFIX_VV(Vec4i, <<);

  DUNE_NSIMD_OPASSIGN_V(Vec4i, >>);
  DUNE_NSIMD_OPINFIX_VV(Vec4i, >>);

  // Vec2q
  DUNE_NSIMD_OPASSIGN_V(Vec2q, /);
  DUNE_NSIMD_OPASSIGN_S(Vec2q, /);
  DUNE_NSIMD_OPINFIX_VV(Vec2q, /);
  DUNE_NSIMD_OPINFIX_VS(Vec2q, /);

  DUNE_NSIMD_OPASSIGN_V(Vec2q, %);
  DUNE_NSIMD_OPASSIGN_S(Vec2q, %);
  DUNE_NSIMD_OPINFIX_VV(Vec2q, %);
  DUNE_NSIMD_OPINFIX_VS(Vec2q, %);

  DUNE_NSIMD_OPASSIGN_V(Vec2q, <<);
  DUNE_NSIMD_OPINFIX_VV(Vec2q, <<);

  DUNE_NSIMD_OPASSIGN_V(Vec2q, >>);
  DUNE_NSIMD_OPINFIX_VV(Vec2q, >>);

  // these are necessary to resolve ambiguous overloads.
  inline Vec4fb operator==(Vec4fb a, Vec4fb b) { return !(a ^ b); }
  inline Vec2db operator==(Vec2db a, Vec2db b) { return !(a ^ b); }
#endif

#if MAX_VECTOR_SIZE >= 256
  // Vec8i
  DUNE_NSIMD_OPASSIGN_V(Vec8i, /);
  DUNE_NSIMD_OPINFIX_SV(Vec8i, /);
  DUNE_NSIMD_OPINFIX_VV(Vec8i, /);
  DUNE_NSIMD_OPINFIX_VS(Vec8i, /);

  DUNE_NSIMD_OPASSIGN_V(Vec8i, %);
  DUNE_NSIMD_OPASSIGN_S(Vec8i, %);
  DUNE_NSIMD_OPINFIX_SV(Vec8i, %);
  DUNE_NSIMD_OPINFIX_VV(Vec8i, %);
  DUNE_NSIMD_OPINFIX_VS(Vec8i, %);

  DUNE_NSIMD_OPASSIGN_V(Vec8i, <<);
  DUNE_NSIMD_OPINFIX_VV(Vec8i, <<);

  DUNE_NSIMD_OPASSIGN_V(Vec8i, >>);
  DUNE_NSIMD_OPINFIX_VV(Vec8i, >>);

  // Vec4q
  DUNE_NSIMD_OPASSIGN_V(Vec4q, /);
  DUNE_NSIMD_OPASSIGN_S(Vec4q, /);
  DUNE_NSIMD_OPINFIX_SV(Vec4q, /);
  DUNE_NSIMD_OPINFIX_VV(Vec4q, /);
  DUNE_NSIMD_OPINFIX_VS(Vec4q, /);

  DUNE_NSIMD_OPASSIGN_V(Vec4q, %);
  DUNE_NSIMD_OPASSIGN_S(Vec4q, %);
  DUNE_NSIMD_OPINFIX_SV(Vec4q, %);
  DUNE_NSIMD_OPINFIX_VV(Vec4q, %);
  DUNE_NSIMD_OPINFIX_VS(Vec4q, %);

  DUNE_NSIMD_OPASSIGN_V(Vec4q, <<);
  DUNE_NSIMD_OPINFIX_VV(Vec4q, <<);

  DUNE_NSIMD_OPASSIGN_V(Vec4q, >>);
  DUNE_NSIMD_OPINFIX_VV(Vec4q, >>);

  // these are necessary to resolve ambiguous overloads.
  inline Vec8fb operator==(Vec8fb a, Vec8fb b) { return !(a ^ b); }
  inline Vec4db operator==(Vec4db a, Vec4db b) { return !(a ^ b); }
#endif // MAX_VECTOR_SIZE >= 256

#if MAX_VECTOR_SIZE >= 512
  // Vec16i
  DUNE_NSIMD_OPASSIGN_V(Vec16i, /);
  DUNE_NSIMD_OPINFIX_SV(Vec16i, /);
  DUNE_NSIMD_OPINFIX_VV(Vec16i, /);
  DUNE_NSIMD_OPINFIX_VS(Vec16i, /);

  DUNE_NSIMD_OPASSIGN_V(Vec16i, %);
  DUNE_NSIMD_OPASSIGN_S(Vec16i, %);
  DUNE_NSIMD_OPINFIX_SV(Vec16i, %);
  DUNE_NSIMD_OPINFIX_VV(Vec16i, %);
  DUNE_NSIMD_OPINFIX_VS(Vec16i, %);

  DUNE_NSIMD_OPASSIGN_V(Vec16i, <<);
  DUNE_NSIMD_OPINFIX_VV(Vec16i, <<);

  DUNE_NSIMD_OPASSIGN_V(Vec16i, >>);
  DUNE_NSIMD_OPINFIX_VV(Vec16i, >>);

  // Vec8q
  DUNE_NSIMD_OPASSIGN_V(Vec8q, /);
  DUNE_NSIMD_OPASSIGN_S(Vec8q, /);
  DUNE_NSIMD_OPINFIX_SV(Vec8q, /);
  DUNE_NSIMD_OPINFIX_VV(Vec8q, /);
  DUNE_NSIMD_OPINFIX_VS(Vec8q, /);

  DUNE_NSIMD_OPASSIGN_V(Vec8q, %);
  DUNE_NSIMD_OPASSIGN_S(Vec8q, %);
  DUNE_NSIMD_OPINFIX_SV(Vec8q, %);
  DUNE_NSIMD_OPINFIX_VV(Vec8q, %);
  DUNE_NSIMD_OPINFIX_VS(Vec8q, %);

  DUNE_NSIMD_OPASSIGN_V(Vec8q, <<);
  DUNE_NSIMD_OPINFIX_VV(Vec8q, <<);

  DUNE_NSIMD_OPASSIGN_V(Vec8q, >>);
  DUNE_NSIMD_OPINFIX_VV(Vec8q, >>);

#if INSTRSET >= 9 // native AVX512
  inline Vec16fb operator&&(Vec16fb a, bool b) { return a && Vec16fb(b); }
  inline Vec16fb operator&&(bool a, Vec16fb b) { return Vec16fb(a) && b; }
  inline Vec16fb operator||(Vec16fb a, bool b) { return a || Vec16fb(b); }
  inline Vec16fb operator||(bool a, Vec16fb b) { return Vec16fb(a) || b; }

  inline Vec16ib operator&&(Vec16ib a, bool b) { return a && Vec16ib(b); }
  inline Vec16ib operator&&(bool a, Vec16ib b) { return Vec16ib(a) && b; }
  inline Vec16ib operator||(Vec16ib a, bool b) { return a || Vec16ib(b); }
  inline Vec16ib operator||(bool a, Vec16ib b) { return Vec16ib(a) || b; }

  inline Vec8db operator&&(Vec8db a, bool b) { return a && Vec8db(b); }
  inline Vec8db operator&&(bool a, Vec8db b) { return Vec8db(a) && b; }
  inline Vec8db operator||(Vec8db a, bool b) { return a || Vec8db(b); }
  inline Vec8db operator||(bool a, Vec8db b) { return Vec8db(a) || b; }

  inline Vec8qb operator&&(Vec8qb a, bool b) { return a && Vec8qb(b); }
  inline Vec8qb operator&&(bool a, Vec8qb b) { return Vec8qb(a) && b; }
  inline Vec8qb operator||(Vec8qb a, bool b) { return a || Vec8qb(b); }
  inline Vec8qb operator||(bool a, Vec8qb b) { return Vec8qb(a) || b; }
#else // INSTRSET < 9, no native AVX512
  // these are necessary to resolve ambiguous overloads.
  inline Vec16fb operator==(Vec16fb a, Vec16fb b) { return !(a ^ b); }
  inline Vec8db  operator==(Vec8db  a, Vec8db  b) { return !(a ^ b); }
  inline Vec16ib operator==(Vec16ib a, Vec16ib b) { return !(a ^ b); }
  inline Vec8qb  operator==(Vec8qb  a, Vec8qb  b) { return !(a ^ b); }
#endif // INSTRSET < 9

  // these seem to have been forgotten
  inline Vec16ib operator!(Vec16i a) { return a == 0; }
  inline Vec8qb  operator!(Vec8q  a) { return a == 0; }
#endif // MAX_VECTOR_SIZE >= 512

#undef DUNE_NSIMD_OPASSIGN_V
#undef DUNE_NSIMD_OPASSIGN_S
#undef DUNE_NSIMD_OPINFIX_VV
#undef DUNE_NSIMD_OPINFIX_VS

#ifdef VCL_NAMESPACE
} // namespace VCL_NAMESPACE
#endif

#endif //DUNE_NSIMD_NSIMD_HH
