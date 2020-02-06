import mpmath as mp
from mpmath import j,pi,exp,jtheta,re,im,sin,sinh,log,atan,tanh,cos,diff,qp


"""
SINGLE SPLITING QUENCH
"""
# conformal map for single splitting quench. zeta=exp(i*theta)
def theta(x,t,a):
    if x>0:
        return atan(-(x-t)/a)
    else:
        return -pi+atan(-(x-t)/a)

def thetaB(x,t,a):
    if x>0:
        return atan(-(x+t)/a)
    else:
        return -pi+atan(-(x+t)/a)

#EE for dirac fermion
def S_single_dirac(x1,x2,t,a):
    t1,t2,tb1,tb2 = theta(x1,t,a), theta(x2,t,a), thetaB(x1,t,a), thetaB(x2,t,a)
    z1,z2,zb1,zb2 = j*exp(j*t1), j*exp(j*t2), j*exp(-j*tb1), j*exp(-j*tb2)

    deriv = a**2 / ( exp(j*(t1-tb1+t2-tb2)/2) * cos(t1)*cos(t2)*cos(tb1)*cos(tb2) )
    conn = (z1-z2)*(zb1-zb2)
    disconn = (z1-zb1)*(z2-zb2)
    cross = (z1-zb2)*(z2-zb1)

    return re(log( deriv * conn * disconn / cross ))/6 - log((x2-x1)**2)/6

#EE for holographic CFT
def S_single_hol_conn(x1,x2,t,a):
    t1,t2,tb1,tb2 = theta(x1,t,a), theta(x2,t,a), thetaB(x1,t,a), thetaB(x2,t,a)
    z1,z2,zb1,zb2 = j*exp(j*t1), j*exp(j*t2), j*exp(-j*tb1), j*exp(-j*tb2)

    deriv = a**2 / ( exp(j*(t1-tb1+t2-tb2)/2) * cos(t1)*cos(t2)*cos(tb1)*cos(tb2) )
    conn = (z1-z2)*(zb1-zb2)

    return re(log( deriv * conn ))/6 - log((x2-x1)**2)/6

def S_single_hol_disconn(x1,x2,t,a):
    t1,t2,tb1,tb2 = theta(x1,t,a), theta(x2,t,a), thetaB(x1,t,a), thetaB(x2,t,a)
    z1,z2,zb1,zb2 = j*exp(j*t1), j*exp(j*t2), j*exp(-j*tb1), j*exp(-j*tb2)

    deriv = a**2 / ( exp(j*(t1-tb1+t2-tb2)/2) * cos(t1)*cos(t2)*cos(tb1)*cos(tb2) )
    disconn = (z1-zb1)*(z2-zb2)

    return re(log( deriv * disconn ))/6 - log((x2-x1)**2)/6

def S_single_hol(x1,x2,t,a):
    t1,t2,tb1,tb2 = theta(x1,t,a), theta(x2,t,a), thetaB(x1,t,a), thetaB(x2,t,a)
    z1,z2,zb1,zb2 = j*exp(j*t1), j*exp(j*t2), j*exp(-j*tb1), j*exp(-j*tb2)

    deriv = a**2 / ( exp(j*(t1-tb1+t2-tb2)/2) * cos(t1)*cos(t2)*cos(tb1)*cos(tb2) )
    conn = (z1-z2)*(zb1-zb2)
    disconn = (z1-zb1)*(z2-zb2)

    return min([re(log( deriv * conn )),re(log( deriv * disconn ))])/6 - log((x2-x1)**2)/6


"""
DOUBLE SPLITING QUENCH
"""
# theta/eta function
def jt1(nu,tau):
    return jtheta(1,pi*nu,exp(pi*j*tau),0)

def jt1d(nu,tau):
    return jtheta(1,pi*nu,exp(pi*j*tau),1)

def eta(tau):
    return exp(j*pi*tau/12)*qp(exp(2*pi*j*tau),exp(2*pi*j*tau))

# conformal map from cylinder to plane, s=beta^{-1}
def w(v,s,b):
    return -j*b*( jt1d(v,j*s)/jt1(v,j*s) + jt1d(v+j*s/2,j*s)/jt1(v+j*s/2,j*s) + j )

def Dw(v,s,b):
    return diff(lambda x: w(x,s,b), v, 1)

def cutoff(s,b):
    return abs(im(w( mp.findroot(lambda y: diff( lambda x: im(w(x-j*s/4,s,b)),y,1 ) ,1/2,tol=10**(-10) ) -j*s/4,s,b)))

# inverse map
def v(x,t,s,b):
    # c is atrificial parameter for findroot.
    c = 10**(-10)
    if x>b:
        return j*mp.findroot(lambda y: re(w(j*y,s,b)-(x-t)), [-s/2+c,0-c],"bisect")
    elif x<-b:
        return j*mp.findroot(lambda y: re(w(j*y,s,b)-(x-t)), [0+c,s/2-c],"bisect")
    else:
        tolerant = 10**(-20)
        normparam = 10**11
        return 1/2+j*mp.findroot(lambda y: tanh(re(w(1/2+j*y,s,b)-(x-t))/normparam), 0,tol=tolerant)

def vb(x,t,s,b):
    if abs(x)>b:
        return -v(x,-t,s,b)
    else:
        return 1-v(x,-t,s,b)

#EE for dirac fermion
def S_double_dirac(x1,x2,t,s,b):
    v1,v2,vb1,vb2 = v(x1,t,s,b),v(x2,t,s,b),vb(x1,t,s,b),vb(x2,t,s,b)

    deriv = ((2*pi)**(-4)) * Dw(v1,s,b)*(-Dw(-vb1,s,b))*Dw(v2,s,b)*(-Dw(-vb2,s,b))
    conn = jt1(v1-v2,j*s) * jt1(vb2-vb1,j*s) / eta(j*s)**6
    disconn = jt1(v1-vb1+j*s/2,j*s) * jt1(v2-vb2+j*s/2,j*s)
    cross = jt1(v1-vb2+j*s/2,j*s) * jt1(v2-vb1+j*s/2,j*s)

    return re(log( deriv * (conn * disconn / cross)**2 ))/12-log((x2-x1)**2)/6

#EE for holographic CFT
def S_double_hol_conn(x1,x2,t,s,b):
    v1,v2,vb1,vb2 = v(x1,t,s,b),v(x2,t,s,b),vb(x1,t,s,b),vb(x2,t,s,b)

    deriv = (pi**(-4)) * Dw(v1,s,b)*(-Dw(-vb1,s,b))*Dw(v2,s,b)*(-Dw(-vb2,s,b))
    if s<1:
        conn = (s**2) * sinh(pi * (v1-v2)/s) * sinh(pi * (vb1-vb2)/s)
    else:
        conn =  sin(pi * (v1-v2)) * sin(pi * (vb1-vb2))

    return re(log( deriv * conn**2 ))/12-log((x2-x1)**2)/6

def S_double_hol_disconn(x1,x2,t,s,b):
    v1,v2,vb1,vb2 = v(x1,t,s,b),v(x2,t,s,b),vb(x1,t,s,b),vb(x2,t,s,b)
    #difference from mirror twist op
    p1,n1,p2,n2 = v1-vb1+j*s/2, v1-vb1-j*s/2, v2-vb2+j*s/2, v2-vb2-j*s/2

    deriv = (pi**(-4)) * Dw(v1,s,b)*(-Dw(-vb1,s,b))*Dw(v2,s,b)*(-Dw(-vb2,s,b))
    if s<1:
        g1 = min([re(sinh(pi * p1/s)**2),re(sinh(pi * n1/s)**2)])
        g2 = min([re(sinh(pi * p2/s)**2),re(sinh(pi * n2/s)**2)])
        disconnSquare = (s**4) * g1 * g2
    else:
        m1 = sin(pi * p1) * sin(pi * p2)
        m2 = sin(pi * p1) * sin(pi * n2) * exp(pi*s)
        m3 = sin(pi * n1) * sin(pi * p2) * exp(pi*s)
        m4 = sin(pi * n1) * sin(pi * n2)
        disconnSquare =  min([re(m1**2),re(m2**2),re(m3**2),re(m4**2)])

    return re(log( deriv * disconnSquare ))/12-log((x2-x1)**2)/6

def S_double_hol(x1,x2,t,s,b):
    v1,v2,vb1,vb2 = v(x1,t,s,b),v(x2,t,s,b),vb(x1,t,s,b),vb(x2,t,s,b)
    p1,n1,p2,n2 = v1-vb1+j*s/2, v1-vb1-j*s/2, v2-vb2+j*s/2, v2-vb2-j*s/2

    deriv = (pi**(-4)) * Dw(v1,s,b)*(-Dw(-vb1,s,b))*Dw(v2,s,b)*(-Dw(-vb2,s,b))
    if s<1:
        conn = (s**2) * sinh(pi * (v1-v2)/s) * sinh(pi * (vb1-vb2)/s)
        g1 = min([re(sinh(pi * p1/s)**2),re(sinh(pi * n1/s)**2)])
        g2 = min([re(sinh(pi * p2/s)**2),re(sinh(pi * n2/s)**2)])
        disconnSquare = (s**4) * g1 * g2
    else:
        conn =  sin(pi * (v1-v2)) * sin(pi * (vb1-vb2))
        m1 = sin(pi * p1) * sin(pi * p2)
        m2 = sin(pi * p1) * sin(pi * n2) * exp(pi*s)
        m3 = sin(pi * n1) * sin(pi * p2) * exp(pi*s)
        m4 = sin(pi * n1) * sin(pi * n2)
        disconnSquare =  min([re(m1**2),re(m2**2),re(m3**2),re(m4**2)])

    return min([re(log( deriv * re(conn)**2 )),re(log( deriv * disconnSquare ))])/12-log((x2-x1)**2)/6

"""
PLOTTING
"""
if __name__ == '__main__':

    mp.dps = 500  #dps for precision
    mp.pretty = True

    """
    plot of DOUBLE SPLITTING QUENCH
    """
    b = 10
    maxtime = 300
    plotpoints = 300

#     s = 0.945
#     x1, x2 = 100, 200
#     mp.plot([lambda t: S_double_dirac(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = 10, 30
#     mp.plot([lambda t: S_double_dirac(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = 30, 100
#     mp.plot([lambda t: S_double_dirac(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = -150, 100
#     mp.plot([lambda t: S_double_dirac(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
    s = 2
#     x1, x2 = 100, 200
#     mp.plot([lambda t: S_double_dirac(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = 10, 30
#     mp.plot([lambda t: S_double_dirac(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = 30, 100
#     mp.plot([lambda t: S_double_dirac(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = -150, 100
#     mp.plot([lambda t: S_double_dirac(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)

#     mp.plot([lambda k: S_double_dirac(-b+k, b-k, 50, s, b)],[0.01,49.9],points=50)
#     x1, x2 = 30,40
#     mp.plot([lambda t: S_double_hol(x1, x2, t, s, b)],[0,50],points=200)

#     ep=1
#     x1, x2 = -b+ep, b-ep
#     mp.plot([lambda t: S_double_dirac(x1, x2, t, s, b)+log((x2-x1)**2)/6],[0,maxtime],points=plotpoints)
#     ep=1
#     x1, x2 = -b-ep, b+ep
#     mp.plot([lambda t: S_double_dirac(x1, x2, t, s, b)],[0,10000],points=plotpoints)

#     s = 0.945
#     plotpoints = 300
#     x1, x2 = 100, 200
#     mp.plot([lambda t: S_double_hol_conn(x1, x2, t, s, b),lambda t: S_double_hol_disconn(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = 10, 30
#     mp.plot([lambda t: S_double_hol_conn(x1, x2, t, s, b),lambda t: S_double_hol_disconn(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = 30, 100
#     mp.plot([lambda t: S_double_hol_conn(x1, x2, t, s, b),lambda t: S_double_hol_disconn(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = -150, 100
#     mp.plot([lambda t: S_double_hol_conn(x1, x2, t, s, b),lambda t: S_double_hol_disconn(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     s = 5.28
#     x1, x2 = 100, 200
#     mp.plot([lambda t: S_double_hol_conn(x1, x2, t, s, b),lambda t: S_double_hol_disconn(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = 10, 30
#     mp.plot([lambda t: S_double_hol_conn(x1, x2, t, s, b),lambda t: S_double_hol_disconn(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = 30, 100
#     mp.plot([lambda t: S_double_hol_conn(x1, x2, t, s, b),lambda t: S_double_hol_disconn(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = -51, 51
#     mp.plot([lambda t: S_double_hol_conn(x1, x2, t, s, b),lambda t: S_double_hol_disconn(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)

    """
    plot of SINGLE SPLITTING QUENCH
    """
#     s,b=3.37,50
    a = cutoff(s,b)
#     print(a)
#     maxtime = 200
#     plotpoints = 1000
#     mp.plot(lambda x: cutoff(x,50)/50,[0.9,10],points=100)

#     x1, x2 = 50, 150
#     mp.plot(lambda t: S_single_dirac(x1, x2, t, a),[0,maxtime],points=plotpoints)
#     x1, x2 = -20, 50
#     mp.plot(lambda t: S_single_dirac(x1, x2, t, a),[0,maxtime],points=plotpoints)
#     x1, x2 = 50, 150
#     mp.plot([lambda t: S_single_hol_conn(x1, x2, t, a),lambda t: S_single_hol_disconn(x1, x2, t, a)],[0,maxtime],points=plotpoints)
#     x1, x2 = -20, 50
#     mp.plot([lambda t: S_single_hol_conn(x1, x2, t, a),lambda t: S_single_hol_disconn(x1, x2, t, a)],[0,maxtime],points=plotpoints)


    """
    difference
    """
    x1=30
    x2=40
#     mp.plot( [lambda t: S_double_dirac(x1, x2, t, s, b)/(S_single_dirac(x1-b, x2-b, t, a)+S_single_dirac(x1+b, x2+b, t, a))], [0,400],points=200 )
    mp.plot( [lambda t: S_double_hol(x1, x2, t, s, b)/(S_single_hol(x1-b, x2-b, t, a)+S_single_hol(x1+b, x2+b, t, a))], [0,100],points=100 )
