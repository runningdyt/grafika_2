//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Arnócz László
// Neptun : PV8AK9
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) {
    int logLen;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0) {
        char * log = new char[logLen];
        int written;
        glGetShaderInfoLog(handle, logLen, &written, log);
        printf("Shader log:\n%s", log);
        delete log;
    }
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK) {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}

// check if shader could be linked
void checkLinking(unsigned int program) {
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK) {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}

// vertex shader in GLSL
const char *vertexSource = R"(
#version 140
precision highp float;

in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
out vec2 texcoord;			// output attribute: texture coordinate

void main() {
    texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
    gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
#version 140
precision highp float;

uniform sampler2D textureUnit;
in  vec2 texcoord;			// interpolated texture coordinates
out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

void main() {
    fragmentColor = texture(textureUnit, texcoord);
}
)";

#include <vector>
#define PI 3.14
const float hibatures = 1e-4;

// row-major matrix 4x4
struct mat4 {
    float m[4][4];
public:
    mat4() {}
    mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }
    
    mat4 operator*(const mat4& right) {
        mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
            }
        }
        return result;
    }
    operator float*() { return &m[0][0]; }
};

struct vec4 {
    float v[4];
    vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
        v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }
    vec4(const vec4& otherVec){
        v[0]=otherVec.v[0]; v[1]=otherVec.v[1]; v[2]=otherVec.v[2]; v[3]=otherVec.v[3];
    }
    vec4 operator*(const mat4& mat)const {
        vec4 result;
        for (int j = 0; j < 4; j++) {
            result.v[j] = 0;
            for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
        }
        return result;
    }
    vec4 operator*(float a) const{
        vec4 res(v[0] * a, v[1] * a, v[2] * a);
        return res;
    }
    vec4 operator/(float a)const {
        vec4 res(v[0] / a, v[1] / a, v[2] / a);
        return res;
    }
    vec4 operator-(float a)	const{
        vec4 res(v[0] - a, v[1] - a, v[2] - a);
        return res;
    }
    vec4 operator+(float a)	const{
        vec4 res(v[0] + a, v[1] + a, v[2] + a);
        return res;
    }
    vec4 operator+(const vec4& other)const {
        vec4 res(v[0] + other.v[0], v[1] + other.v[1], v[2] + other.v[2],v[3]+other.v[3]);
        return res;
    }
    vec4 operator-(const vec4& other)const {
        vec4 res(v[0] - other.v[0], v[1] - other.v[1], v[2] - other.v[2], v[3]-other.v[3]);
        return res;
    }
    vec4 operator*(const vec4& other) const{
        vec4 res(v[0] * other.v[0], v[1] * other.v[1], v[2] * other.v[2], v[3]*other.v[3]);
        return res;
    }
    vec4 operator/(const vec4& other) const{
        vec4 res(v[0] / other.v[0], v[1] / other.v[1], v[2] / other.v[2], v[3]/other.v[3]);
        return res;
    }
    vec4& operator+=(const vec4& other) {
        v[0] += other.v[0]; v[1] += other.v[1]; v[2] += other.v[2]; v[3]+=other.v[3];
        return *this;
    }
    vec4 cross(const vec4& other)const{
        vec4 res(v[1] * other.v[2] - v[2] * other.v[1], v[2] * other.v[0] - v[0] * other.v[2],
                 v[0] * other.v[1] - v[1] * other.v[0], 0);
        return res;
    }
    vec4 normalize() const{
        return *this * (1.0 / Length());
    }
    float dot(const vec4& other)const {
        return (v[0] * other.v[0] + v[1] * other.v[1] + v[2] * other.v[2]);
    }
    float Length() const { return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    vec4 RodriguesRotation(vec4 unitvector, float theta){
        float x = unitvector.v[0];
        float y = unitvector.v[1];
        float z = unitvector.v[2];
        float t = (theta/180) * PI;
        mat4 R(	(cosf(t)+x*x*(1-cosf(t))), 		(z*sinf(t)+x*y*(1-cosf(t))), 	(-y*sinf(t)+x*z*(1-cosf(t))),	0,
               (x*y*(1-cosf(t))-z*sin(t)), 	(cosf(t)+y*y*(1-cos(t))), 		(x*sinf(t)+y*z*(1-cosf(t))),	0,
               (y*sinf(t)+x*z*(1-cosf(t))), 	(-x*sinf(t)+y*z*(1-cos(t))), 	(cosf(t)+z*z*(1-cosf(t)))	,	0,
               0,0,0,1);
        return (*this * R);
    }
    void print(){
        printf("x:%f	y:%f	z:%f \n", this->v[0], this->v[1], this->v[2]);
    }
};

struct Ray {
    vec4 p0;
    vec4 dv;
    Ray(vec4 p0, vec4 dv){
        this -> p0 = p0;
        this -> dv = dv;
    }
};

struct Light {
    vec4 Lout, type, dir;
    Light(vec4 Lout, vec4 dir){
        this-> Lout = Lout;
        this-> dir = dir.normalize();
    }
};

struct Material{
    vec4 ka, kd, ks, n, k;
    vec4 F0;
    float shineFactor;
    float n_toresmutato;
    bool isReflective = false; // tükröző felület?
    bool isRefractive = false; // törő felület?
    Material(vec4 n, vec4 k, vec4 ka, vec4 kd, vec4 ks, float shineFactor){
        this->ka = ka;
        this->kd = kd;
        this->ks = ks;
        this->n = n;
        this->k = k;
        this->shineFactor = shineFactor;
        F0 = (((n-1)*(n-1))+(k*k))/(((n+1)*(n+1))+(k*k));
    }
    void setReflective(){
        isReflective = true;
    }
    void setRefractive(){
        isRefractive = true;
    }
    vec4 shade(vec4 normal, vec4 incomingRayDirection, vec4 incomingLightDirection, vec4 IncomingLightRadiance){
        normal = normal.normalize();
        float cosTheta = normal.dot(incomingLightDirection);
        //printf("%f\n",cosTheta);
        if (cosTheta < 0)
            return vec4(0,0,0,0);
        if (ks.v[0] == 0 && ks.v[1] == 0 && ks.v[2] == 0){
            return IncomingLightRadiance * kd * cosTheta;
        }
        vec4 normalref = IncomingLightRadiance * kd * cosTheta;
        vec4 halfwayVector = (incomingRayDirection + normal).normalize();
        float cosDelta = normal.dot(halfwayVector);
        if (cosDelta<0)
            return normalref;
        vec4 phonblin = IncomingLightRadiance * ks * pow(cosDelta, shineFactor);
        vec4 r(normalref + phonblin);
        return r;
    }
    vec4 reflect(vec4 incomingRayDirection, vec4 normal) {
        vec4 retval((incomingRayDirection - normal) * ((normal.dot(incomingRayDirection)) * 2.0f));
        return retval;
    }
    vec4 refract(vec4 incomingRayDirection, vec4 normal) {
        float ior = n_toresmutato;
        float cosa = (normal.dot(incomingRayDirection))*-1.0;
        if (cosa < 0) { cosa = -cosa; normal = normal*-1; ior = 1/n_toresmutato; }
        float disc = 1 - (1 - cosa * cosa)/ior/ior;
        if (disc < 0) return reflect(incomingRayDirection, normal);
        return incomingRayDirection/ior + normal * (cosa/ior - sqrt(disc));
    }
    vec4 Fresnel(vec4 normal, vec4 incomingRayDirection){
        float cosa = fabs(normal.dot(incomingRayDirection));
        vec4 retval = F0 + (vec4(1, 1, 1, 1) - F0) * pow(cosa,5);
        return retval;
    }
};

struct Hit{
    float t;
    vec4 position;
    vec4 normal;
    Material* material;
    Hit() { t = -1; }
    Hit(const Hit& otherHit){
        t=otherHit.t;
        normal=otherHit.normal;
        material=otherHit.material;
    }
};

struct Intersectable{
    Material* material;
    void addMaterial(Material* m){
        material = m;
    }
    virtual Hit intersect(const Ray& ray)=0;
    virtual vec4 getNormal(const vec4& intersect) = 0; // a felületi normálist adott pontban lekérdező függvény
};



class Sphere : public Intersectable {
    vec4 center;
    float radius;
public:
    Sphere(vec4 c, float r){
        center=c;
        radius=r;
    }
    Hit intersect(const Ray& ray) {
        Hit retval;
        retval.material = material;
        
        float a = ray.dv.dot(ray.dv);
        float b = (ray.p0-center).dot(ray.dv)*2;
        float c = ((ray.p0-center).dot(ray.p0-center))-(radius*radius);
        
        float d = b * b - 4 * a * c;
        if(d < 0)	//prevent getting rekt
            retval.t = -1.0;
        else{
            float t1 = (-1.0 * b - sqrt(b * b - 4 * a * c)) / 2.0 * a;
            float t2 = (-1.0 * b + sqrt(b * b - 4 * a * c)) / 2.0 * a;
            if (t1<t2)
                retval.t = t1;
            else
                retval.t = t2;
        }
        if (fabs(retval.t) < hibatures)
            retval.t = -1;
        retval.position = ray.p0 + ray.dv * retval.t;
        retval.normal = getNormal(ray.p0 + ray.dv * retval.t);
        return retval;
    }
    vec4 getNormal(const vec4& intersect){
        //derivativex = 2 * (intersect.v[0] - central.v[0]);
        //derivativey = 2 * (intersect.v[1] - central.v[1]);
        //derivativez = 2 * (intersect.v[2] - central.v[2]);
        vec4 retval = (intersect - center) * 2;
        return retval.normalize();
    }
};



class Triangle : public Intersectable {
    
public:
    vec4 r1,r2,r3;
    Triangle(vec4 r1, vec4 r2, vec4 r3){
        this -> r1 = r1;
        this -> r2 = r2;
        this -> r3 = r3;
    }
    Triangle(const Triangle& otherTriangle){
        *this = otherTriangle;
    }
    Triangle& operator=(const Triangle& otherTriangle){
        this -> r1 = otherTriangle.r1;
        this -> r2 = otherTriangle.r2;
        this -> r3 = otherTriangle.r3;
        this -> material = otherTriangle.material;
        return *this;
    }
    Hit intersect(const Ray& ray){
        Hit retval;
        retval.material = material;
        vec4 null(0,0,0,0);
        retval.normal = getNormal(null);		//don't need that
        vec4 n = retval.normal;
        float t = ((r1 - ray.p0).dot(n))/(ray.dv.dot(n));
        if (t<0)
            return retval;
        vec4 p = ray.p0+ray.dv*t;
        if ((((r2 - r1).cross(p - r1)).dot(n)) > 0 &&
            (((r3 - r2).cross(p - r2)).dot(n)) > 0 &&
            (((r1 - r3).cross(p - r3)).dot(n)) > 0 ){
            retval.position=p;
            retval.t=t;
            retval.normal=n;
            return retval;
        }
        else return retval;
    }
    vec4 getNormal(const vec4& intersect){
        return ((r2-r1).cross(r3-r1)).normalize();
    }
    void Translate(float x, float y, float z){
        mat4 M(	1,0,0,1,
               0,1,0,1,
               0,0,1,1,
               x,y,z,1);
        
        r1 = r1*M;
        r2 = r2*M;
        r3 = r3*M;
    };
};

//fű

class Plane : public Intersectable {
    
public:
    vec4 r1, normal;
    Plane(vec4 r1, vec4 normal){
        this -> r1 = r1;
        this -> normal = normal.normalize();
    }
    Hit intersect(const Ray& ray){
        Hit retval;
        retval.material = material;
        retval.normal = normal;
        float t = ((r1 - ray.p0).dot(normal))/(ray.dv.dot(normal));
        if (t<0)
            return retval;
        vec4 p = ray.p0+ray.dv*t;
        retval.position = p;
        retval.t = t;
        return retval;
        }
    
    vec4 getNormal(const vec4& intersect){
        return normal.normalize();
    }
};

class PlaneWithHole : public Intersectable {
public:
    vec4 r1, normal, min, max;
    PlaneWithHole(vec4 r1, vec4 normal, vec4 min, vec4 max){
        this -> r1 = r1;
        this -> normal = normal.normalize();
    }
    Hit intersect(const Ray& ray){
        Hit retval;
        retval.material = material;
        retval.normal = normal;
        float t = ((r1 - ray.p0).dot(normal))/(ray.dv.dot(normal));
        if (t<0)
            return retval;
        vec4 p = ray.p0+ray.dv*t;
        if (min.v[0] < p.v[0] && p.v[0] < max.v[0] && min.v[2] < p.v[2] && p.v[2] < max.v[2])
            return retval;

        retval.position = p;
        retval.t = t;
        return retval;
    }
    
    vec4 getNormal(const vec4& intersect){
        return normal.normalize();
    }
};


//ezt majd kibuggolom, szerintem valamelyik oldalon nem clockwise adtam meg
//mert clockwise de eszrevettem hogy az nem lesz jo ezert kiolvastam oket visszafele a tombbol... savage

class Cube : public Intersectable{
    float size;
    vec4 center;
    std::vector<Triangle*> triangles;		//the faces
    int indices[36] = {
        0,1,2,
        0,2,3,
        0,3,4,
        0,4,5,
        5,1,0,
        5,6,1,
        1,6,7,
        1,7,2,
        4,3,2,
        4,2,7,
        6,5,7,
        5,4,7
    };
    std::vector<vec4> vertices;				//the vertices
public:
    Cube(float size, vec4 center){
        this->center = center;
        this->size = size;
        float d = sqrtf(3)/2 * size;
        float x = center.v[0];
        float y = center.v[1];
        float z = center.v[2];
        
        vertices.push_back(vec4 (x-d,y+d,z+d));//a
        vertices.push_back(vec4 (x+d,y+d,z+d));//b
        vertices.push_back(vec4 (x+d,y-d,z+d));//c
        vertices.push_back(vec4 (x-d,y-d,z+d));//d
        vertices.push_back(vec4 (x-d,y-d,z-d));//e
        vertices.push_back(vec4 (x-d,y+d,z-d));//f
        vertices.push_back(vec4 (x+d,y+d,z-d));//g
        vertices.push_back(vec4 (x+d,y-d,z-d));//h
        
        for (int i=0; i<36; i+=3){
            Triangle* t = new Triangle(vertices[indices[i+2]], vertices[indices[i+1]], vertices[indices[i]]);
            t->addMaterial(material);
            triangles.push_back(t);
        }
    }
    Cube(const Cube& otherCube){
        *this = otherCube;
    }
    Cube& operator=(const Cube& otherCube){
        this->center = otherCube.center;
        this->size = otherCube.size;
        this->material = otherCube.material;
        for(Triangle* t : triangles){
            delete t;
        }
        for(int i=0; i<12; i++){
            triangles[i] = new Triangle(*otherCube.triangles[i]);
        }
        return *this;
    }
    void addMaterial(Material* m){
        for (Triangle* t : triangles) {
            t->addMaterial(m);
        }
    }
    vec4 getNormal(const vec4& intersect){
        vec4 retval(0,0,0,0);
        return retval;
    }
    
    Hit intersect(const Ray& ray){
            Hit bestHit;
            for(Triangle* obj : triangles) {
                Hit hit = obj->intersect(ray); //  hit.t < 0 if no intersection
                if(hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) 	bestHit = hit;
            }
            return bestHit;
    }
    void Translate(float x, float y, float z){
        for(Triangle* obj : triangles){
            obj->Translate(x,y,z);
        }
    }
    void Rotate(vec4 unitvector, float degree){
        vec4 newcenter = center.RodriguesRotation(unitvector, degree);
        Cube newcube(this->size, newcenter);
        newcube.addMaterial(material);
        *this = newcube;
        }
    ~Cube(){
        for(Triangle* t : triangles){
            delete t;
        }
    }
};

class Ellipsoid : public Intersectable {
    float a, b, c;
    vec4 center;
public:
    Ellipsoid(vec4 cent, float a, float b, float c){
        center = cent;
        this ->c = c;
        this ->b = b;
        this ->a = a;
    }
    Hit intersect(const Ray& ray) {
        Hit retval;
        retval.material = material;
        vec4 rayorigin = ray.p0 - center;
        vec4 raynormal = ray.dv.normalize();
        
        float a_ = ((raynormal.v[0]*raynormal.v[0])/(a*a))
        + ((raynormal.v[1]*raynormal.v[1])/(b*b))
        + ((raynormal.v[2]*raynormal.v[2])/(c*c));
        
        float b_ = ((2*rayorigin.v[0]*raynormal.v[0])/(a*a))
        + ((2*rayorigin.v[1]*raynormal.v[1])/(b*b))
        + ((2*rayorigin.v[2]*raynormal.v[2])/(c*c));
        
        float c_ = ((rayorigin.v[0]*rayorigin.v[0])/(a*a))
        + ((rayorigin.v[1]*rayorigin.v[1])/(b*b))
        + ((rayorigin.v[2]*rayorigin.v[2])/(c*c))
        - 1;
        
        float d = b_ * b_ - 4 * a_ * c_;
        if(d < 0)	//prevent getting rekt
            retval.t = -1.0;
        else{
            float t1 = (-1.0 * b_ - sqrt(b_ * b_ - 4 * a_ * c_)) / 2.0 * a_;
            float t2 = (-1.0 * b_ + sqrt(b_ * b_ - 4 * a_ * c_)) / 2.0 * a_;
            if (t1<t2)
                retval.t = t1;
            else
                retval.t = t2;
        }
        if (retval.t < hibatures)
            retval.t = 0;
        retval.normal = getNormal(ray.p0 + ray.dv * retval.t);
        return retval;
    }
    vec4 getNormal(const vec4& intersect){
        float derivativex = 2.0 * (intersect.v[0] - center.v[0])/a;
        float derivativey = 2.0 * (intersect.v[1] - center.v[1])/b;
        float derivativez = 2.0 * (intersect.v[2] - center.v[2])/c;
        vec4 ret = vec4(derivativex, derivativey, derivativez).normalize();
        return ret;
    }
};

// Színtér objektum
class Scene{
public:
    std::vector<Intersectable*> objs;
    std::vector<Light*> lights;
    vec4 La{0.5,0.9,0.9};    //sky intensity
    int maxdepth=6;
    
    void addObject(Intersectable* newobj){
        objs.push_back(newobj);
    }
    void addLight(Light* l){
        lights.push_back(l);
    }
    Hit firstIntersect(Ray& ray) {
        Hit bestHit;
        for(Intersectable* obj : objs) {
            Hit hit = obj->intersect(ray); //  hit.t < 0 if no intersection
            if(hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) 	bestHit = hit;
        }
        return bestHit;
    }
    
    int sign(float a){
        if (a<0)
            return -1;
        return 1;
    }
    
    vec4 Trace(Ray& ray, int depth){
        if (depth > maxdepth)
            return La;
        Hit hit = firstIntersect(ray);
        if(hit.t < 0) return La; // nothing
        vec4 outRadiance = hit.material->ka * La;
        for(Light* l : lights ){
            Ray shadowRay(hit.position + hit.normal*hibatures*sign(hit.normal.dot((ray.dv*(-1)).normalize())),
                          (l->dir*-1).normalize());
            Hit shadowHit = firstIntersect(shadowRay);
            if(shadowHit.t < 0 || shadowHit.t > 10000)
                outRadiance += hit.material->shade(hit.normal, (ray.dv).normalize(), (l->dir*-1).normalize(), l->Lout);
        }
        if(hit.material->isReflective){
            vec4 reflectionDir = (hit.material->reflect(ray.dv, hit.normal)).normalize();
            Ray reflectedRay(hit.position + hit.normal*hibatures*sign(hit.normal.dot((ray.dv*-1).normalize())), reflectionDir);
            outRadiance += Trace(reflectedRay, depth+1) * hit.material->Fresnel((ray.dv*-1).normalize(), hit.normal);
        }
        if(hit.material->isRefractive) {
            vec4 refractionDir = (hit.material->refract(ray.dv,hit.normal)).normalize();
            Ray refractedRay(hit.position - hit.normal*hibatures*sign(hit.normal.dot((ray.dv*(-1)).normalize())), refractionDir);
            outRadiance += Trace(refractedRay,depth+1)*(vec4(1,1,1,0)-hit.material->Fresnel((ray.dv*-1).normalize(), hit.normal));
        }
        return outRadiance;
    }
};

struct MyCamera600x600{
    vec4 eyePosition, centerOfPlanePosition, direction,  upUnitVector, rightUnitVector;
    float FOV;
    int XM,YM;
    MyCamera600x600(vec4 eyePosition, vec4 centerOfPlanePosition){
        this -> eyePosition = eyePosition;
        this -> centerOfPlanePosition = centerOfPlanePosition;
        upUnitVector.v[1] = centerOfPlanePosition.v[1]+2;
        direction = centerOfPlanePosition - eyePosition;
        rightUnitVector = direction.cross(upUnitVector);
        upUnitVector = rightUnitVector.cross(direction);
        rightUnitVector = rightUnitVector.normalize();
        upUnitVector = upUnitVector.normalize();
        direction = direction.normalize();
        XM = windowWidth;
        YM = windowHeight;
    }
    MyCamera600x600(vec4 eyePosition, vec4 target, float FOV){
        this -> eyePosition = eyePosition;
        this -> FOV = FOV;
        float radian = tanf((FOV*M_PI/180)/2);
        float focalDistance = (1)/(radian);
        centerOfPlanePosition = eyePosition + ((target - eyePosition).normalize())*focalDistance;
        upUnitVector.v[1] = centerOfPlanePosition.v[1]+2;
        direction = centerOfPlanePosition - eyePosition;
        rightUnitVector = direction.cross(upUnitVector);
        upUnitVector = rightUnitVector.cross(direction);
        rightUnitVector = rightUnitVector.normalize();
        upUnitVector = upUnitVector.normalize();
        direction = direction.normalize();
        XM = windowWidth;
        YM = windowHeight;
    }
    MyCamera600x600(){
        eyePosition.v[0]=0; eyePosition.v[1]=0; eyePosition.v[2]=0;
        centerOfPlanePosition.v[0]=0; centerOfPlanePosition.v[1]=0; centerOfPlanePosition.v[2]=-1;
        direction = centerOfPlanePosition - eyePosition;	//(0,0,-1) by default
        upUnitVector.v[0] = 0; upUnitVector.v[1] = 1; upUnitVector.v[2] = 0;
        rightUnitVector.v[0] = 1; rightUnitVector.v[1] = 0; rightUnitVector.v[2] = 0; //hardcoding
        XM = windowWidth;
        YM = windowHeight;
    }
    Ray GetRay(int x, int y){
        vec4 p = centerOfPlanePosition + rightUnitVector * ((2*float(x)/XM)-1) + upUnitVector * ((2*float(y)/YM)-1);
        //p.print();
        Ray retval(eyePosition, (p-eyePosition).normalize());
        return retval;
    }
    void Translate(float x, float y, float z){
        mat4 M(	1,0,0,1,
               0,1,0,1,
               0,0,1,1,
               x,y,z,1 );
        
        eyePosition = eyePosition * M;
        centerOfPlanePosition = centerOfPlanePosition * M;
        direction = direction * M;
        //upUnitVector = upUnitVector * M; rightUnitVector = rightUnitVector * M;
    }
    void Rotate(vec4 unit, float theta){
        //eyePosition = eyePosition.RodriguesRotation(unit, theta);
        centerOfPlanePosition = centerOfPlanePosition.RodriguesRotation(unit, theta);
        direction = direction.RodriguesRotation(unit, theta);
        upUnitVector = upUnitVector.RodriguesRotation(unit, theta);
        rightUnitVector = rightUnitVector.RodriguesRotation(unit, theta);
    }
};

// handle of the shader program
unsigned int shaderProgram;

class FullScreenTexturedQuad {
    unsigned int vao, textureId;	// vertex array object id and texture id
public:
    void Create(vec4 image[windowWidth * windowHeight]) {
        glGenVertexArrays(1, &vao);	// create 1 vertex array object
        glBindVertexArray(vao);		// make it active
        
        unsigned int vbo;		// vertex buffer objects
        glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects
        
        // vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        static float vertexCoords[] = { -1, -1,   1, -1,  -1, 1,
            1, -1,   1,  1,  -1, 1 };	// two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
        // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,	0, NULL);     // stride and offset: it is tightly packed
        
        // Create objects by setting up their vertex data on the GPU
        glGenTextures(1, &textureId);  				// id generation
        glBindTexture(GL_TEXTURE_2D, textureId);    // binding
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, image); // To GPU
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    
    void Draw() {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        int location = glGetUniformLocation(shaderProgram, "textureUnit");
        if (location >= 0) {
            glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
        }
        glDrawArrays(GL_TRIANGLES, 0, 6);	// draw two triangles forming a quad
    }
};

// The virtual world: single quad
FullScreenTexturedQuad fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    
    Scene scene;
    
    //a red material for the unit sphere
    Material spherematerialRed(vec4(0,0,0),vec4(0,0,0),
                            vec4(0.8, 0, 0),vec4(0.8, 0.5, 0.5),vec4(0.5, 0.5, 0.5),1.8974f);
    
    //my materials
    Material goldMaterial(vec4(0.17,0.35,1.5), vec4(3.1,2.7,1.9), vec4(), vec4(), vec4(), 10);
    goldMaterial.setReflective();
    Material grassMaterial(vec4(), vec4(), vec4(0.2, 0.3, 0.2), vec4(0.1, 0.5, 0.2),vec4(), 0);
   
    
    Material cubematerial(vec4(0,0,0),vec4(0,0,0),
                            vec4(0.2, 0.2, 0.8),vec4(0.8, 0.5, 0.5),vec4(1, 1, 1),500.8974f);
    Material silverMaterial(vec4(0.14,0.16,0.13),vec4(4.1,2.3,3.1),vec4(), vec4(), vec4(), 10);
    silverMaterial.setReflective();
    /*
     n/k az r,g,b hullámhosszain: 0.17/3.1, 0.35/2.7, 1.5/1.9), felületének implicit egyenlete nem lineáris (pl. kvadratikus), a másik ezüstözött (n/k az r,g,b hullámhosszain: 0.14/4.1, 0.16/2.3, 0.13/3.1) poligonháló.
     */
    
    
    vec4 SunDirection(-0.212527, -0.898794, -0.383408);
    Light Sun(vec4(1,1,1), SunDirection);
    scene.addLight(&Sun);
    
    MyCamera600x600 mycam(vec4(0,0,10), vec4(0,-0.9,8));
    //MyCamera600x600 mycam(MyCamera600x600(vec4(0,0,10), vec4(0,-3,0), 50));
    Sphere unitSphere(vec4(0,0,0),1);             //az egyseggömb, ami a 0,0,0 ban helyezkedik el
    unitSphere.addMaterial(&spherematerialRed);
    Sphere mysphere(vec4(-3,-3,0),0.5);
    mysphere.addMaterial(&silverMaterial);
    Ellipsoid myellipsoid(vec4(0,-2,0),0.7,1,1);
    myellipsoid.addMaterial(&goldMaterial);
    Plane myplane(vec4(0,-4.5,0), vec4(0,1,0));
    myplane.addMaterial(&grassMaterial);
    PlaneWithHole myplanewithhole(vec4(0,-4.5,0), vec4(0,1,0),vec4(-4,-4.5,-10),vec4(4,-4.5,0));
    myplanewithhole.addMaterial(&grassMaterial);
    Sphere s(vec4(-2,-1,0), 1);
    s.addMaterial(&goldMaterial);
    Cube mycube(0.5, vec4(-2,-3,0));
    mycube.addMaterial(&spherematerialRed);
    //mycube.Rotate(vec4(0,0,0), 10);
    mycam.Rotate(vec4(1,0,0), 0);//  ezt használhatod forgatásra, bár most az egész tengelyt forgatja
    //mycam.Translate(<#float x#>, <#float y#>, <#float z#>) //"transzlál"
    scene.addObject(&unitSphere);
    //scene.addObject(&mysphere);
    //scene.addObject(&myplane);
    scene.addObject(&myplanewithhole);
    scene.addObject(&mycube);
    scene.addObject(&s);
    //scene.addObject(&myellipsoid);
    
    static vec4 background[windowWidth * windowHeight];
    for (int x = 0; x < windowWidth; x++) {
        for (int y = 0; y < windowHeight; y++) {
            Ray newray = mycam.GetRay(x,y);
            background[y * windowWidth + x] = scene.Trace(newray,4);
        }
    }
    
    
    fullScreenTexturedQuad.Create( background );
    
    // Create vertex shader from string
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    if (!vertexShader) {
        printf("Error in vertex shader creation\n");
        exit(1);
    }
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    checkShader(vertexShader, "Vertex shader error");
    
    // Create fragment shader from string
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (!fragmentShader) {
        printf("Error in fragment shader creation\n");
        exit(1);
    }
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader, "Fragment shader error");
    
    // Attach shaders to a single program
    shaderProgram = glCreateProgram();
    if (!shaderProgram) {
        printf("Error in shader program creation\n");
        exit(1);
    }
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    
    // Connect Attrib Arrays to input variables of the vertex shader
    glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
    
    // Connect the fragmentColor to the frame buffer memory
    glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory
    
    // program packaging
    glLinkProgram(shaderProgram);
    checkLinking(shaderProgram);
    // make this program run
    glUseProgram(shaderProgram);
}

void onExit() {
    glDeleteProgram(shaderProgram);
    printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);							// background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    fullScreenTexturedQuad.Draw();
    glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
    if (key == 'q') exit(0);
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
    
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {// GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
        //printf("x:%d , y:%d\n", pX, pY );
    }
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char * argv[]) {
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
    glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow(argv[0]);
    
#if !defined(__APPLE__)
    glewExperimental = true;	// magic
    glewInit();
#endif
    
    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    
    onInitialization();
    
    glutDisplayFunc(onDisplay);                // Register event handlers
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);
    
    glutMainLoop();
    onExit();
    return 1;
}
