#pragma once

#include <iostream>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include "parlay/parallel.h"
#include "parlay/primitives.h"

namespace pargeo {

  struct _empty {
    int arr[0]; // todo this produces a struct of size 0 but seems dangerous, need to check
    _empty(){}
    
    template<class T>
    _empty(T x){}

    friend bool operator<(_empty a, _empty b) {return true;}
    friend bool operator==(_empty a, _empty b) {return true;}
  };

  template <int _dim, class _tData, class _tFloat, class _tAtt>
  class _point {

    static constexpr _tData empty = std::numeric_limits<_tData>::max();

  public:

    static constexpr int dim = _dim;
    static constexpr bool hasAttribute = !std::is_same<_empty, _tAtt>::value;
    static constexpr int data_len = _dim + hasAttribute; 
    static constexpr _tFloat eps = 1e-5;
    using floatT = _tFloat;
    using attT = _tAtt;

    _tData x[_dim];
    _tAtt attribute;

    _point() { for (int i=0; i<_dim; ++i) x[i]=empty; }

    _point(_tData* p) { for (int i=0; i<_dim; ++i) x[i]=p[i]; }

    _point(_tData* p, _tAtt _attribute): attribute(_attribute) {
      for (int i=0; i<_dim; ++i) x[i]=p[i];
    }

    _point(_point* p): attribute(p->attribute) { for (int i=0; i<_dim; ++i) x[i]=p->x[i]; }

    template<class _tIn>
    _point(parlay::slice<_tIn*,_tIn*> p) {
      for(int i=0; i<_dim; ++i) x[i] = (_tData)p[i];}

    _point(parlay::sequence<_tData>& p, int s, int e){
      if(e-s+1 > _dim){ // has attribute
        attribute = (_tAtt) p[e-1];
      }
      for(int i=0;i<_dim;++i) x[i] = p[s+i];
    }

    void setEmpty() {x[0]=empty;}

    bool isEmpty() const {return x[0]==empty;}

    template<class pt>
    pt operator+(pt op2) const {
      _tData xx[_dim];
      for (int i=0; i<_dim; ++i) xx[i] = x[i]+op2.x[i];
      return pt(xx, attribute);}

    template<class pt> // todo add template for all (for derived class)
    pt operator-(pt op2) const {
      _tData xx[_dim];
      for (int i=0; i<_dim; ++i) xx[i] = x[i]-op2.x[i];
      return pt(xx, attribute);}

    _point operator*(_tData dv) const {
      _tData xx[_dim];
      for (int i=0; i<_dim; ++i) xx[i] = x[i]*dv;
      return _point(xx, attribute);}

    _point operator/(_tData dv) const {
      _tData xx[_dim];
      for (int i=0; i<_dim; ++i) xx[i] = x[i]/dv;
      return _point(xx, attribute);}

    _tData& operator[](int i) {return x[i];}

    _tData operator[](int i) const {return x[i];}

    _tData& at(int i) {return x[i];}

    friend bool operator<(const _point& a,const _point& b) {
      _tData aVal = 0;
      _tData bVal = 0;
      for (int ii=0; ii<dim; ++ii) {
        aVal += a[ii];
        bVal += b[ii];
      }
      return aVal < bVal;
    }

    static bool attComp(const _point& a, const _point& b) {
      return a.attribute < b.attribute;
    }

	static bool attCompRev(const _point& a, const _point& b) {
      return b.attribute < a.attribute;
    }

	  static _point max_point() {
		  _point ret;
		  for(int i=0;i<dim;++i) ret[i]=std::numeric_limits<_tData>::max()-1;
		  return ret;
	  }

	  static _point min_point() {
		  _point ret;
		  for(int i=0;i<dim;++i) ret[i]=std::numeric_limits<_tData>::lowest()+1;
		  return ret;
	  }

    friend bool operator==(const _point& a, const _point& b) {
      for (int ii=0; ii<dim; ++ii) {
	     if (a[ii] != b[ii]) return false;}
       if(a.attribute == b.attribute) return true; else return false;
    }

    friend bool operator!=(const _point& a, const _point& b) {return !(a==b);}

    _tData* coords() {return x;}

    _tData distSqr(const _point& p) const {
      _tData xx=0;
      for (int i=0; i<_dim; ++i) xx += (x[i]-p.x[i])*(x[i]-p.x[i]);
      return xx;}

    inline _tFloat dist(const _point& p) const {
      return sqrt(distSqr(p));
    }

    _tData dot(const _point& p2) const {
      _tData r = 0;
      for(int i=0; i<dim; ++i) r += x[i]*p2[i];
      return r;}

    _point mult(_tData c) {
      _point r;
      for(int i=0; i<dim; ++i) r[i] = x[i]*c;
      return r;}

    _tData lenSqr() {
      _tData xx=0;
      for (int i=0; i<_dim; ++i) xx += x[i]*x[i];
      return xx;}

    _tFloat length() {
      return sqrt((_tFloat)lenSqr());}
  };

  template<int dim>
  using point = _point<dim, double, double, _empty>;

  template<int dim>
  using fpoint = _point<dim, float, float, _empty>;

  template<int dim>
  using lpoint = _point<dim, long, double, _empty>;

  template<int dim, class _tAtt>
  using pointD = _point<dim, double, double, _tAtt>;

  template<class _A, class _B>
  _B pointCast(_B p) {
    _B q;
    for (int i=0; i<p.dim; ++i) q[i] = p[i];
    return q;
  }
}

template<int dim>
static std::ostream& operator<<(std::ostream& os, const pargeo::point<dim>& v) {
  for (int i=0; i<v.dim; ++i)
    os << v.x[i] << " ";
  return os;
}

template<int dim>
static std::ostream& operator<<(std::ostream& os, const pargeo::fpoint<dim>& v) {
  for (int i=0; i<v.dim; ++i)
    os << v.x[i] << " ";
  return os;
}

template<int dim>
static std::ostream& operator<<(std::ostream& os, const pargeo::lpoint<dim>& v) {
  for (int i=0; i<v.dim; ++i)
    os << v.x[i] << " ";
  return os;
}

template<int dim, class _tAtt>
static std::ostream& operator<<(std::ostream& os, const pargeo::pointD<dim, _tAtt>& v) {
  for (int i=0; i<v.dim; ++i)
    os << v.x[i] << " ";
  return os;
}
