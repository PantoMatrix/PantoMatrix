bm_v = new THREE.MeshPhongMaterial({
    color: 0x08519c,
    emissive: 0x08306b,
    specular: 0x08519c,
    shininess: 10,
    side: THREE.DoubleSide
});

jm_v = new THREE.MeshPhongMaterial({
    color: 0x08306b,
    emissive: 0x000000,
    specular: 0x111111,
    shininess: 90,
    side: THREE.DoubleSide
});

bm_a = new THREE.MeshPhongMaterial({
    color: 0x980043,
    emissive: 0x67001f,
    specular: 0x6a51a3,
    shininess: 10,
    side: THREE.DoubleSide
});

jm_a = new THREE.MeshPhongMaterial({
    color: 0x67001f,
    emissive: 0x000000,
    specular: 0x111111,
    shininess: 90,
    side: THREE.DoubleSide
});

bm_b = new THREE.MeshPhongMaterial({
    color: 0x3f007d,
    emissive: 0x3f007d,
    specular: 0x807dba,
    shininess: 2,
    side: THREE.DoubleSide
});

jm_b = new THREE.MeshPhongMaterial({
    color: 0x3f007d,
    emissive: 0x000000,
    specular: 0x807dba,
    shininess: 90,
    side: THREE.DoubleSide
});

//------------------


jointmaterial = new THREE.MeshLambertMaterial({
    color: 0xc57206,
    emissive: 0x271c18,
    side: THREE.DoubleSide,
    // shading: THREE.FlatShading,
    wireframe: false,
    shininess: 90,
});

bonematerial = new THREE.MeshPhongMaterial({
    color: 0xbd9a6d,
    emissive: 0x271c18,
    side: THREE.DoubleSide,
    // shading: THREE.FlatShading,
    wireframe: false
});

jointmaterial2 = new THREE.MeshPhongMaterial({
    color: 0x1562a2,
    emissive: 0x000000,
    specular: 0x111111,
    shininess: 30,
    side: THREE.DoubleSide
});

bonematerial2 = new THREE.MeshPhongMaterial({
    color: 0x552211,
    emissive: 0x882211,
    // emissive: 0x000000,
    specular: 0x111111,
    shininess: 30,
    side: THREE.DoubleSide
});

bonematerial3 = new THREE.MeshPhongMaterial({
    color: 0x176793,
    emissive: 0x000000,
    specular: 0x111111,
    shininess: 90,
    side: THREE.DoubleSide
});



jointmaterial4 = new THREE.MeshPhongMaterial({
    color: 0xFF8A00,
    emissive: 0x000000,
    specular: 0x111111,
    shininess: 90,
    side: THREE.DoubleSide
});


bonematerial4 = new THREE.MeshPhongMaterial({
    color: 0x53633D,
    emissive: 0x000000,
    specular: 0xFFC450,
    shininess: 90,
    side: THREE.DoubleSide
});



bonematerial44 = new THREE.MeshPhongMaterial({
    color: 0x582A72,
    emissive: 0x000000,
    specular: 0xFFC450,
    shininess: 90,
    side: THREE.DoubleSide
});

jointmaterial5 = new THREE.MeshPhongMaterial({
    color: 0xAA5533,
    emissive: 0x000000,
    specular: 0x111111,
    shininess: 30,
    side: THREE.DoubleSide
});

bonematerial5 = new THREE.MeshPhongMaterial({
    color: 0x552211,
    emissive: 0x772211,
    specular: 0x111111,
    shininess: 30,
    side: THREE.DoubleSide
});


markermaterial = new THREE.MeshPhongMaterial({
    color: 0xc57206,
    emissive: 0x271c18,
    side: THREE.DoubleSide,
    // shading: THREE.FlatShading,
    wireframe: false,
    shininess: 20,
});

markermaterial2 = new THREE.MeshPhongMaterial({
    color: 0x1562a2,
    emissive: 0x271c18,
    side: THREE.DoubleSide,
    // shading: THREE.FlatShading,
    wireframe: false,
    shininess: 20,
});

markermaterial3 = new THREE.MeshPhongMaterial({
    color: 0x555555,
    emissive: 0x999999,
    side: THREE.DoubleSide,
    // shading: THREE.FlatShading,
    wireframe: false,
    shininess: 20,
});


var makeMarkerGeometry_Sphere10 = function(markerName, scale) {
    return new THREE.SphereGeometry(10, 60, 60);
};

var makeMarkerGeometry_Sphere3 = function(markerName, scale) {
    return new THREE.SphereGeometry(3, 60, 60);
};

var makeMarkerGeometry_SphereX = function(markerName, scale) {
    return new THREE.SphereGeometry(5, 60, 60);
};

var makeJointGeometry_SphereX = function(X) {
    return function(jointName, scale) {
        return new THREE.SphereGeometry(X, 60, 60);
    };    
};


var makeJointGeometry_Sphere1 = function(jointName, scale) {
    return new THREE.SphereGeometry(2 / scale, 60, 60);
};

var makeJointGeometry_Sphere2 = function(jointName, scale) {
    return new THREE.SphereGeometry(1 / scale, 60, 60);
};

var makeJointGeometry_Dode = function(jointName, scale) {
    return new THREE.DodecahedronGeometry(1 / scale, 0);
};

var makeBoneGeometry_Cylinder1 = function(joint1Name, joint2Name, length, scale) {
    return new THREE.CylinderGeometry(1.5 / scale, 0.7 / scale, length, 40);
};

var makeBoneGeometry_Cylinder2 = function(joint1Name, joint2Name, length, scale) {
    // if (joint1Name.includes("LeftHip"))
    //     length = 400;
    return new THREE.CylinderGeometry(1.5 / scale, 0.2 / scale, length, 40);   
};

var makeBoneGeometry_Cylinder3 = function(joint1Name, joint2Name, length, scale) {
    var c1 = new THREE.CylinderGeometry(1.5 / scale, 0.2 / scale, length / 1, 20);
    var c2 = new THREE.CylinderGeometry(0.2 / scale, 1.5 / scale, length / 1, 40);

    var material = new THREE.MeshPhongMaterial({
        color: 0xF7FE2E
    });
    var mmesh = new THREE.Mesh(c1, material);
    mmesh.updateMatrix();
    c2.merge(mmesh.geometry, mmesh.matrix);
    return c2;
};

var makeBoneGeometry_Box1 = function(joint1Name, joint2Name, length, scale) {
    return new THREE.BoxGeometry(1 / scale, length, 1 / scale, 40);
};


var makeJointGeometry_Empty = function(jointName, scale) {
    return new THREE.SphereGeometry(0.001, 60, 60);
};

var makeBoneGeometry_Empty = function(joint1Name, joint2Name, length, scale) {
    return new THREE.CylinderGeometry(0.001, 0.001, 0.001, 40);
};
