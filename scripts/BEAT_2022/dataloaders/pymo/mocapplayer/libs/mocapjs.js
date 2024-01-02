/*!
 * The MIT License (MIT)
 * 
 * Copyright (c) 2016 Omid Alemi
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 */
/******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId])
/******/ 			return installedModules[moduleId].exports;
/******/
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			exports: {},
/******/ 			id: moduleId,
/******/ 			loaded: false
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.loaded = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(0);
/******/ })
/************************************************************************/
/******/ ([
/* 0 */
/***/ function(module, exports, __webpack_require__) {

	BVHCharacter = __webpack_require__(1);
	C3DCharacter = __webpack_require__(5);
	MocapParsers = __webpack_require__(2);

/***/ },
/* 1 */
/***/ function(module, exports, __webpack_require__) {

	var parsers = __webpack_require__(2);
	
	var BVHCharacter = BVHCharacter || {};
	
	
	BVHCharacter = function(n, jm, bm, jg, bg) {
	    this.name = n;
	
	    this.jointMaterial = jm;
	    this.boneMaterial = bm;
	    this.makeJointGeometryFCN = jg;
	    this.makeBoneGeometryFCN = bg;
	
	    this.bvh = [];
	    this.skeleton = new THREE.Group();
	
	    this.skelScale = 1;
	    this.jointMeshes = [];
	    this.boneMeshes = [];
	    this.rootMeshes = [];
	
	    this.originPosition = new THREE.Vector3(0, 0, 0);
	
	    this.ready = false;
	    this.frameTime = 1 / 30;
	    this.frameCount = 0;
	    this.animIndex = 0;
	    this.animStartTimeRef = 0;
	    this.animOffset = 0;
	    this.playing = true;
	
	    this.debug = true;
	    this.useWorker = true;
	
	    this.webSocket = [];
	    this.streamProtocol = "BVHStream";
	    this.keepStreamedFrames = true;
	    this.isStreaming = false;
	
	    var self = this;
	
	    //
	
	    this.log = function(m) {
	        if (self.debug)
	            console.log(self.name + ": " + m.toString());
	    };
	
	    this.loadFromURL = function(url, callback) {
	        self.log("Loading the mocap file ...");
	        //Pace.start();
	        reader = new parsers.bvhParser(this.name + "READER");
	        this.url = url;
	        reader.load(url, self.createSkel, self.fillFrames);
	
	        this.callb = callback;
	    };
	
	    this.fillFrames = function() {
	        // self.log("Ready!");
	        self.ready = true;
	        self.playing = true;
	
	        if (self.callb)
	            self.callb();
	    }
	
	    this.createSkel = function(data) {
	        self.bvh = data;
	        self.frameCount = data.frameCount;
	        self.frameTime = data.frameTime;
	
	        self.log("Mocap file loaded.");
	
	        self.log("Creating the WebGL Joints.");
	        self.buildSkelJoints(self.bvh.getSkeleton(), 0);
	
	        self.log("Creating the WebGL Bones.");
	        self.buildSkelBones(self.jointMeshes[0]);
	        
	        self.skeleton.add(self.jointMeshes[0]);
	        self.setSkeletonScale(self.skelScale);
	        self.setSkelUp();
	    };
	
	
	    // Beginning of the Stream Code
	    this.onHeaderReceived = function(data) {
	        self.log("Loading the mocap header (skeleton) from the stream...");
	        headerReader = new parsers.bvhStreamParser();
	        headerReader.readHeader(data, self.createSkel);
	
	        if (self.callb)
	            self.callb();
	
	        Pace.stop();
	    }
	
	    this.onDataChunckReceived = function(rawFrames) {
	        var aa = [];
	
	        for (f = 1; f < rawFrames.length; f++) {
	            var parts = rawFrames[f].trim().split(" ");
	            for (var j = 0; j < parts.length; j++)
	                parts[j] = +parts[j];
	            aa.push(parts);
	        }
	        diff = self.bvh.fillFrameArray(aa);
	        self.frameCount = self.bvh.frameArray.length;
	
	
	        if (!self.playing) {
	            self.animStartTimeRef = Date.now();
	            //  self.animOffset -= rawFrames.length;
	        }
	        /*
	        // else
	        // self.animOffset = self.animIndex;
	        if (diff > 0)
	        	self.animOffset -= rawFrames.length + 1;
	        // self.animIndex -= rawFrames.length; //math.max(0,math.min(rawFrames.length, self.bvh.bufferSize));
	        */
	        self.fillFrames();
	        Pace.stop();
	    }
	
	    this.loadFromStream = function(url, callback) {
	        self.log("Connecting to the stream server...");
	        self.isStreaming = true;
	        this.callb = callback;
	        self.webSocket = new WebSocket(url);
	
	        self.webSocket.onerror = function(event) {
	            self.log("Error connecting to the stream server " + event.origin);
	        };
	
	        self.webSocket.onopen = function(event) {
	            self.log("Connected to the stream server " + event.origin);
	            Pace.stop();
	        };
	
	        self.webSocket.onmessage = function(event) {
	            // I'm not doing much of a type and content checking here. Let's just trust the sender for now!
	            // Protocol for header:
	            // $HEADER$
	            // BVH...
	            // Protocl for data chunk with id#:
	            // $FRAMES$id#$
	
	            var messageLines = event.data.split('\n');
	
	            // self.log("Received somthing!");
	            // self.log("The first line is : " + messageLines[0]);
	
	            if (messageLines.length < 1)
	                return;
	
	            if (messageLines[0] == "$HEADER$") {
	                self.onHeaderReceived(event.data);
	
	            } else if (messageLines[0].startsWith("$FRAMES$")) {
	                chunckID = parseInt(messageLines[0].split("$")[2]);
	                self.onDataChunckReceived(messageLines, chunckID);
	            }
	        };
	
	    };
	
	    this.requestFrames = function(i) {
	        self.webSocket.send("$GETFRAMES" + i + "$");
	    }
	
	    // End of the Stream Code
	
	    this.setOriginPosition = function(x, y, z) {
	        self.originPosition.set(x, y, z);
	    };
	
	    this.setSkeletonScale = function(s) {
	        self.rootMeshes.forEach(function(c) {
	            c.scale.set(s, s, s);
	        });
	        self.jointMeshes[0].scale.set(s, s, s);
	        self.jointMeshes[0].position.multiplyScalar(s);
	    };
	
	    this.buildSkelJoints = function(joint, parent) {
	        var jointMesh = new THREE.Mesh(self.makeJointGeometryFCN(joint.name, self.skelScale), self.jointMaterial);
	        jointMesh.bvhIndex = joint.jointIndex;
	        jointMesh.offsetVec = new THREE.Vector3(joint.offset[0], joint.offset[1], joint.offset[2]);
	        jointMesh.name = joint.name;
	        jointMesh.jointparent = parent;
	        var a, b, c;
	        if (!joint.isEndSite()) {
	            a = joint.channelNames[joint.channelNames.length - 3][0];
	            b = joint.channelNames[joint.channelNames.length - 2][0];
	            c = joint.channelNames[joint.channelNames.length - 1][0];
	        }
	        jointMesh.rotOrder = a + b + c;
	        self.jointMeshes.push(jointMesh);
	        
	        jointMesh.position.set(jointMesh.offsetVec.x, jointMesh.offsetVec.y, jointMesh.offsetVec.z);
	        
	        // var axisHelper = new THREE.AxisHelper( 10 / self.skelScale );
	        // jointMesh.add( axisHelper );
	
	
	        joint.children.forEach(function(child) {
	            jointMesh.add(self.buildSkelJoints(child, 1));
	        });
	
	        return jointMesh;
	    };
	
	    this.buildSkelBones = function(rootJointMesh) {
	        rootJointMesh.traverse(function(childJointMesh) {            
	            if (childJointMesh.parent !== null)
	            {
	                if (typeof childJointMesh.bvhIndex === "undefined")
	                    return;
	                // move origin (.translate)
	                // rotate
	                // translate (offset + position)
	                h = math.abs(childJointMesh.offsetVec.length());
	                var bgeometry = self.makeBoneGeometryFCN(childJointMesh.parent.name, childJointMesh.name, h, self.skelScale);
	                
	                //BEGIN - Universal      
	                if (childJointMesh.offsetVec.y !== 0)                                               
	                    // bgeometry.translate(0, Math.sign(childJointMesh.offsetVec.y) * h / 2, 0);
	                    bgeometry.translate(0, -h/2, 0);
	                else                                
	                    bgeometry.translate(0, -h / 2,  0);
	                    
	                               
	                dx = Math.atan2(childJointMesh.offsetVec.z,childJointMesh.offsetVec.y);                  
	                dy = Math.atan2(childJointMesh.offsetVec.x,childJointMesh.offsetVec.z);
	                dz = Math.atan2(childJointMesh.offsetVec.x,childJointMesh.offsetVec.y);   
	   
	   
	                osx = math.sign(childJointMesh.offsetVec.x) === 0 ? 0: math.sign(childJointMesh.offsetVec.x);
	                osy = math.sign(childJointMesh.offsetVec.y) === 0 ? 0: math.sign(childJointMesh.offsetVec.y);
	                osz = math.sign(childJointMesh.offsetVec.z) === 0 ? 0: math.sign(childJointMesh.offsetVec.z);
	
	                osxy = math.sign(childJointMesh.offsetVec.x) === 0 ? 0: math.sign(childJointMesh.offsetVec.y);
	                osyx = math.sign(childJointMesh.offsetVec.y) === 0 ? 0: math.sign(childJointMesh.offsetVec.x);
	                osyz = math.sign(childJointMesh.offsetVec.y) === 0 ? 0: math.sign(childJointMesh.offsetVec.z);
	                oszy = math.sign(childJointMesh.offsetVec.z) === 0 ? 0: math.sign(childJointMesh.offsetVec.y);
	
	                
	                if (osz <0)
	                    bgeometry.rotateZ(1*(math.pi-dz));
	                else if (osz === 0)
	                    bgeometry.rotateZ(1*(math.pi-dz));
	                    // console.log();  
	                else if (osz > 0)
	                    bgeometry.rotateZ(1*(2*math.pi-dz));
	
	
	                if (oszy >0)
	                    bgeometry.rotateX(-1 *(2*math.pi-dx));
	                else if (childJointMesh.offsetVec.z === 0)
	                    // bgeometry.rotateX(-1*(math.pi-dx));
	                    console.log();
	                else if (oszy < 0)
	                    bgeometry.rotateX(-1*(2*math.pi-dx));
	                                                            
	                //  bgeometry.rotateY(math.pi-dy);
	                                                  
	                //END - Universal
	                
	                var boneMesh = new THREE.Mesh(bgeometry, self.boneMaterial);                
	                
	                boneMesh.joint = childJointMesh.parent;
	                boneMesh.name = childJointMesh.parent.name + " > " + childJointMesh.name;
	                
	                childJointMesh.parent.add(boneMesh);                               
	                self.boneMeshes.push(boneMesh);                
	            }
	        });      
	    };
	
	    this.animFrame = function(frame) {
	        var torad = Math.PI / 180;
	        
	        if (frame >= self.frameCount) {
	            self.playing = false;
	            return;
	        }
	        
	
	        this.jointMeshes[0].traverse(function(joint) {
	            
	            if (typeof joint.bvhIndex === "undefined") {
	                return;
	            }
	                
	
	            var bj = self.bvh.jointArray[joint.bvhIndex];
	            var offsetVec = joint.offsetVec;
	            
	            var thisEuler = [];
	
	
	            thisEuler = new THREE.Euler(
	                (bj.channels[frame][bj.rotationIndex.x] * torad),
	                (bj.channels[frame][bj.rotationIndex.y] * torad),
	                (bj.channels[frame][bj.rotationIndex.z] * torad), joint.rotOrder);
	
	
	            joint.localRotMat = new THREE.Matrix4();
	            joint.localRotMat.makeRotationFromEuler(thisEuler);
	            joint.rotation.setFromRotationMatrix(joint.localRotMat);
	
	            if (joint.jointparent !== 0) {
	                // joint.position.set(offsetVec.x, offsetVec.y, offsetVec.z);
	            } else { // root
	                joint.position.set(
	                    bj.channels[frame][bj.positionIndex.x] * self.skelScale + self.originPosition.x,
	                    bj.channels[frame][bj.positionIndex.y] * self.skelScale + self.originPosition.y,
	                    bj.channels[frame][bj.positionIndex.z] * self.skelScale + self.originPosition.z);
	            }
	        });
	
	        if (self.isStreaming) {
	            self.bvh.consumeFrames(frame);
	            self.frameCount = self.bvh.frameArray.length;
	            // console.log(self.frameCount);
	            if (self.frameCount <= 0)
	                self.playing = false;
	
	            self.animOffset = 0; // self.animOffset - frame;
	            self.animStartTimeRef = Date.now();
	        }
	    };
	
	    this.setSkelUp = function() {
	        this.jointMeshes[0].traverse(function(joint) {
	            if (typeof joint.bvhIndex === "undefined")
	                return;
	
	            var bj = self.bvh.jointArray[joint.bvhIndex];
	
	            var offsetVec = joint.offsetVec;
	            var torad = Math.PI / 180;
	            var thisEuler = [];
	
	            thisEuler = new THREE.Euler(0, 0, 0, joint.rotOrder);
	
	            joint.localRotMat = new THREE.Matrix4();
	            joint.localRotMat.makeRotationFromEuler(thisEuler);
	            joint.rotation.setFromRotationMatrix(joint.localRotMat);
	
	            if (joint.jointparent !== 0) {
	                // joint.position.set(offsetVec.x, offsetVec.y, offsetVec.z);
	            } else { // root
	                joint.position.set(self.originPosition.x, self.originPosition.y, self.originPosition.z);
	            }
	        });
	    };
	};
	
	
	module.exports = BVHCharacter;

/***/ },
/* 2 */
/***/ function(module, exports, __webpack_require__) {

	module.exports ={
	    bvhParser: __webpack_require__(3),
	    bvhStreamParser: __webpack_require__(4)
	};

/***/ },
/* 3 */
/***/ function(module, exports) {

	// By Ankit
	var BVHReader = function () {
	    this.load = function (url, callbackHeader, callbackFrameArray) {
	        $.get(url, function (str) {
	            
	            var dataReturn = parse(str);
	
	            
	            var jointStack = dataReturn[0];
	            var jointMap = dataReturn[1];
	            var jointArray = dataReturn[2];
	            var connectivityMatrix = dataReturn[3]
	            _bvh = new BVHReader.BVH.Skeleton(jointStack[0], jointMap, jointArray, dataReturn[3], dataReturn[4], dataReturn[5], []);
	            
	            if (callbackHeader)
	                callbackHeader(_bvh,'BVH');
	            console.log("Blah");
	            _bvh.fillFrameArray(dataReturn[6]);
	
	            if (callbackFrameArray)
	                   callbackFrameArray();
	
	        });
	    };
	
	    function parse(str) {
	        var lines = str.split('\n');
	        var jointStack = [];
	        var jointMap = {};
	        var jointArray = [];
	        var connectivityMatrix = [];
	        var frameCount, frameTime, frameArray = [];
	        var i = 0;
	        //parse structure
	        for (i = 1; i < lines.length; i++) {
	            if (!parseLine(lines[i], jointStack, jointMap, jointArray, connectivityMatrix)) {
	                break;
	            }
	        }
	
	        for (i = i + 1; i < lines.length; i++) {
	            var line = lines[i].trim();
	            //when encountering last line
	            if (line === "")
	                break;
	            if (line.indexOf("Frames") === 0) {
	                frameCount = +(line.split(/\b/)[2]);
	            } else if (line.indexOf("Frame Time") === 0) {
	                frameTime = +( line.substr(line.indexOf(":") + 1).trim() )
	            } else {
	                var parts = line.split(" ");
	                for (var j = 0; j < parts.length; j++)
	                    parts[j] = +parts[j];
	                frameArray.push(parts);
	            }
	        }
	
	        //parse motion
	        return [jointStack, jointMap, jointArray, connectivityMatrix, frameCount, frameTime, frameArray];
	    }
	
	    //parses individual line in the bvh file.
	    var parseLine = function (line, jointStack, jointMap, jointArray, connectivityMatrix) {
	        line = line.trim();
	        if (line.indexOf("ROOT") > -1 || line.indexOf("JOINT") > -1 || line.indexOf("End") > -1) {
	            var parts = line.split(" ");
	            var title = parts[1]; //temporary variable to be used after creating the joint object
	            parts[1] = parts[1] + "-" + jointArray.length;
	            var joint = new BVHReader.BVH.Joint(parts[1]);
	            joint.title = title;
	            jointStack.push(joint);
	
	            joint.jointIndex = Object.keys(jointMap).length;
	            jointMap[parts[1]] = joint;
	            jointArray.push(joint);
	            //if the joint is not an end site
	            if( line.indexOf("End") != 0 ){
	                if (jointArray.length == 1) {
	                    joint.channelOffset = 0;
	                } else {
	                    joint.channelOffset = jointArray[jointArray.length - 2].channelOffset + jointArray[jointArray.length - 2].channelLength;
	                }
	            }else{
	                //channelLength is 0 for end joints
	                joint.channelLength = 0;
	                joint.channelOffset = jointArray[jointArray.length - 2].channelOffset + jointArray[jointArray.length - 2].channelLength;
	            }
	
	        } else if (line.indexOf("{") === 0) {
	
	        } else if (line.indexOf("OFFSET") === 0) {
	            var parts = line.split(" ");
	            jointStack[jointStack.length - 1]["offset"] = parts.slice(1);
	            for(x in jointStack[jointStack.length - 1]["offset"]){
	                jointStack[jointStack.length - 1]["offset"][x] = +jointStack[jointStack.length - 1]["offset"][x]
	            }
	        } else if (line.indexOf("CHANNELS") === 0) {
	            var parts = line.split(" ");
	            jointStack[jointStack.length - 1].setChannelNames(parts.slice(2));
	            jointStack[jointStack.length - 1]["channelLength"] = +parts[1];
	        } else if (line.indexOf("}") === 0) {
	            if (jointStack.length > 1) {
	                child = jointStack.pop();
	                jointStack[jointStack.length - 1].children.push(child);
	                child.parent = jointStack[jointStack.length - 1];
	
	                connectivityMatrix.push([child.parent, child])
	
	                // if(!connectivityMatrix[child.name]){
	                //     connectivityMatrix[child.name] = {}
	                // }
	                // connectivityMatrix[child.name][child.parent.name] = 1;
	
	                // if(!connectivityMatrix[child.parent.name]){
	                //     connectivityMatrix[child.parent.name] = {}
	                // }
	                // connectivityMatrix[child.parent.name][child.name] = 1;
	            }
	        } else if (line.indexOf("MOTION") == 0) {
	            return false;
	        }
	
	        return true;
	    };
	};
	
	BVHReader.BVH = BVHReader.BVH || {};
	
	BVHReader.BVH.Joint = function (name, index) {
	
	    this.name = name;
	    this.children = [];
	    this.isEndSite = function () {
	        return this.children.length == 0;
	    };
	    this.rotationIndex = {};
	    this.positionIndex = {};
	
	    this.getChannels = function () {
	        var allChannels = [];
	        for (i = 0; i < this.skeleton.frameArray.length; i++) {
	            allChannels.push(this.getChannelsAt(i));
	        }
	        return allChannels;
	    };
	    this.getChannelsAt = function (frameNum) {
	        var channelsAtFrame = this.skeleton.frameArray[frameNum];
	        return channelsAtFrame.slice(this.channelOffset, this.channelOffset + this.channelLength);
	    };
	
	    this.setChannelNames = function (nameArr){
	        this.channelNames = nameArr;
	        for(i in this.channelNames){
	            var name = this.channelNames[i];
	            switch(name){
	                case "Xposition": this.positionIndex.x = i; break;
	                case "Yposition": this.positionIndex.y = i; break;
	                case "Zposition": this.positionIndex.z = i; break;
	
	                case "Xrotation": this.rotationIndex.x = i; break;
	                case "Yrotation": this.rotationIndex.y = i; break;
	                case "Zrotation": this.rotationIndex.z = i; break;
	            }
	        }
	    }
	};
	
	BVHReader.BVH.Skeleton = function (root, map, arr, connectivityMatrix, frameCount, frameTime, frameArray) {
	    thisSkeleton = this;
	    this.root = root;
	    this.jointMap = map;
	    this.jointArray = arr;
	    this.connectivityMatrix = connectivityMatrix;
	    this.frameCount = frameCount;
	    this.frameTime = frameTime;
	    this.frameArray = frameArray;
	
	    for (i = 0; i < this.jointArray.length; i++) {
	        this.jointArray[i].skeleton = thisSkeleton;
	    }
	
	
	
	    this.fillFrameArray = function (fa) {
	        this.frameArray = fa;
	        this.frameCount = fa.length;
	            //all the structures are ready. let's calculate the positions
	    for(j=0; j < this.jointArray.length; j++){
	        var joint = this.jointArray[j];
	        updateWithPositions(joint);
	    }
	    }
	
	    this.getChannels = function () {
	        return frameArray;
	    };
	    this.getChannelsAt = function (frameNum) {
	    	//How do I know which column is what?
	        //Why do you need the column index?
	        return frameArray[frameNum];
	    };
	    this.getFrameRate = function () {
	        return frameCount / frameTime;
	    };
	    this.getSkeleton = function () {
	        return root;
	    };
	
	    this.getHeadJoint = function () {
	    	// do a quick search in the joint names to see if any of them matches head, else return the something!!!!
	        return jointMap["Head"];
	    };
	    this.getPositionsAt = function (frameNum) {
	    	//for each joint, calculate its position in XYZ
	        //return an array of joints, each with .x, .y, and .z properties
	    	posFrame = [];
	
	    	for (j=0;j<this.jointArray.length;j++) {
	    		posFrame.push(this.jointArray[j].positions[frameNum]);
	    	}
	
	    	posFrame = posFrame.map(function(d) {
				return {
					x : d[0],
					y : d[1],
					z : d[2],
				};
			});
	
	        return posFrame;
	    };
	    this.getTPose = function () {
	    	// This function is basically the same as the getPositionsAt except that all the rotations will be 0
	        console.log("Not yet implemented");
	    };
	
	    function updatePositions(rootOffset, removeRoot, orientation, camera) {
	      //TODO: compelte the specification of this
	
	      for(j=0; j < this.jointArray.length; j++){
	          var joint = this.jointArray[j];
	          updateWithPositions(joint);
	      }
	    }
	
	    function updateWithPositions(joint){
	        var channelNames = joint.channelNames;
	        joint.channels = joint.getChannels();
	        joint.rotations = [];
	        joint.positions = [];
	        joint.rotmat = [];
	        for(i in joint.channels){
	            var channel = joint.channels[i];
	            var xpos = channel[joint.positionIndex.x] || 0,
	            ypos =  channel[joint.positionIndex.y] || 0,
	            zpos =  channel[joint.positionIndex.z] || 0,
	            xangle =  deg2rad(channel[joint.rotationIndex.x] || 0),
	            yangle =  deg2rad(channel[joint.rotationIndex.y] || 0),
	            zangle= deg2rad(channel[joint.rotationIndex.z] || 0);
	
	            // var rotMatrix = math.transpose(getRotationMatrix(xangle, yangle, zangle, "xyz"));
	            // var rotMatrix = getRotationMatrix1(xangle, yangle, zangle, "xyz"); //this also works
	            var posMatrix = [xpos, ypos, zpos];
	
	            if(joint.parent){
	            	  posMatrix = [0,0,0];  //At least for the bvhs that we have, this should be set to 0
	
	                // var t = vectorAdd(joint.offset, posMatrix);
	                // var u = matrixMultiply(t, joint.parent.rotations[i]);
	
	                // joint.positions[i] = vectorAdd(u, joint.parent.positions[i]);
	                // joint.rotations[i] = matrixMultiply( rotMatrix, joint.parent.rotations[i]);
	                // joint.rotmat[i] = rotMatrix; 
	
	                 if (i==0 && (joint.name ==  "Spine" || joint.name == "L_Femur")) {
	                    /*console.log("head's rot mat: ");
	                    console.log(joint.rotations[i]);
	                    console.log(t);
	                    console.log(u);
	
	                    console.log("x: "+xangle + "y: "+yangle + "z: "+zangle );
	                    console.log(posMatrix);
	                    */
	                }
	
	            }else{
	                //its the root
	                // joint.rotations[i] = rotMatrix;
	                // joint.rotmat[i] = rotMatrix;
	                joint.positions[i] = posMatrix;//vectorAdd(joint.offset , posMatrix);
	                // ^ we can safely ignore the root's offset
	            }
	        }
	    }
	
	    function deg2rad(deg){
	        return deg * (Math.PI/180);
	    }
	
	
	    function getRotationMatrix(alpha, beta, gamma) {
	
	    //inputs are the intrinsic rotation angles in RADIANTS
	    var ca = Math.cos(alpha),
	    	sa = Math.sin(alpha),
	
	    	cb = Math.cos(beta),
	    	sb = Math.sin(beta),
	
	    	cg = Math.cos(gamma),
	    	sg = Math.sin(gamma),
	
	    Rx = [[1, 0, 0], [0, ca, -sa], [0, sa, ca]];
	
	    Ry = [[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]];
	
	    Rz = [[cg, -sg, 0], [sg, cg, 0], [0,    0,   1]];
	
	
	
	
	    var Rzm = math.matrix(Rz);
	    var Rym = math.matrix(Ry);
	    var Rxm = math.matrix(Rx);
	
	    var tt = math.multiply(Rzm, Rym);
	
	    return  math.multiply(tt,Rxm).toArray();
	    //rotationMatrix = math. //Rz*Ry*Rx;
	
	    //     R = Rx*Ry*Rz;
		}
	
	    function getRotationMatrix1 (xangle, yangle, zangle, order){
	        var c1 = Math.cos(xangle),
	        c2 = Math.cos(yangle),
	        c3 = Math.cos(zangle),
	        s1 = Math.sin(xangle),
	        s2 = Math.sin(yangle),
	        s3 = Math.sin(zangle);
	
	        if(order === undefined || order.trim() === ""){
	            order = "zxy";
	        }
	
	        var rotMat = [
	            [1,0,0],
	            [0,1,0],
	            [0,0,1]
	        ];
	
	        switch(order){
	            case "___zxy":
	                rotMat = [
	                    [c2*c3-s1*s2*s3, c2*s3+s1*s2*c3, -s2*c1],
	                    [-c1*s3, c1*c3, s1],
	                    [s2*c3+c2*s1*s3, s2*s3-c2*s1*c3, c2*c1]
	                ];
	            break;
	            default:
	              for (o in order){
	                var axis = order[o];
	                var t;
	                switch(axis){
	                    case "x":
	                        t = [
	                                [1, 0, 0],
	                                [0,  c1, s1],
	                                [0, -s1, c1],
	                            ]
	                        break;
	                    case "y":
	                        t = [
	                                [c2,0,-s2],
	                                [0,1,0],
	                                [s2,0,c2]
	                            ]
	                        break;
	                    case "z":
	                        t = [[c3,s3,0],[-s3,c3,0],[0,0,1]]
	                        break;
	                }
	
	                rotMat = matrixMultiply(t, rotMat)
	              }
	          }
	
	        return rotMat;
	    }
	};
	
	function vectorAdd(a, b){
	    return math.add(math.matrix(a), math.matrix(b)).toArray();
	}
	
	function matrixMultiply(m1, m2) {
	    var a = math.matrix(m1);
	    var b = math.matrix(m2);
	    return math.multiply(a, b).toArray();
	}
	
	
	module.exports = BVHReader;

/***/ },
/* 4 */
/***/ function(module, exports) {

	// BVH parser by Ankit
	// Stream by Omid
	
	
	var BVHStreamParser = function () {
	    this.readHeader = function (str, callback) {
	            var dataReturn = parseHeader(str);
	            var jointStack = dataReturn[0];
	            var jointMap = dataReturn[1];
	            var jointArray = dataReturn[2];
	            var connectivityMatrix = dataReturn[3]
	            if (callback)
	                callback(new BVHStreamParser.BVH.Skeleton(jointStack[0], jointMap, jointArray, dataReturn[3], 0, dataReturn[5], dataReturn[6]),'BVH');
	    };
	
	    function parseHeader(str) {
	        var lines = str.split('\n');
	        var jointStack = [];
	        var jointMap = {};
	        var jointArray = [];
	        var connectivityMatrix = [];
	        var frameCount, frameTime, frameArray = [];
	        var i = 0;
	        //parse structure
	        for (i = 2; i < lines.length; i++) { //  start from 2 to skip the $HEADER$ command
	            if (!parseLine(lines[i], jointStack, jointMap, jointArray, connectivityMatrix)) {
	                break;
	            }
	        }
	
	        for (i = i + 1; i < lines.length; i++) {
	            var line = lines[i].trim();
	            //when encountering last line
	            if (line === "")
	                break;
	            if (line.indexOf("Frames") === 0) {
	                frameCount = +(line.split(/\b/)[2]);
	            } else if (line.indexOf("Frame Time") === 0) {
	                frameTime = +( line.substr(line.indexOf(":") + 1).trim() )
	            } else { /// maybe this should be removed
	                var parts = line.split(" ");
	                for (var j = 0; j < parts.length; j++)
	                    parts[j] = +parts[j];
	                frameArray.push(parts);
	            }
	        }
	
	        //parse motion
	        return [jointStack, jointMap, jointArray, connectivityMatrix, frameCount, frameTime, frameArray];
	    }
	
	    //parses individual line in the bvh file.
	    var parseLine = function (line, jointStack, jointMap, jointArray, connectivityMatrix) {
	        line = line.trim();
	        if (line.indexOf("ROOT") > -1 || line.indexOf("JOINT") > -1 || line.indexOf("End") > -1) {
	            var parts = line.split(" ");
	            var title = parts[1]; //temporary variable to be used after creating the joint object
	            parts[1] = parts[1] + "-" + jointArray.length;
	            var joint = new BVHStreamParser.BVH.Joint(parts[1]);
	            joint.title = title;
	            jointStack.push(joint);
	
	            joint.jointIndex = Object.keys(jointMap).length;
	            jointMap[parts[1]] = joint;
	            jointArray.push(joint);
	            //if the joint is not an end site
	            if( line.indexOf("End") != 0 ){
	                if (jointArray.length == 1) {
	                    joint.channelOffset = 0;
	                } else {
	                    joint.channelOffset = jointArray[jointArray.length - 2].channelOffset + jointArray[jointArray.length - 2].channelLength;
	                }
	            }else{
	                //channelLength is 0 for end joints
	                joint.channelLength = 0;
	                joint.channelOffset = jointArray[jointArray.length - 2].channelOffset + jointArray[jointArray.length - 2].channelLength;
	            }
	
	        } else if (line.indexOf("{") === 0) {
	
	        } else if (line.indexOf("OFFSET") === 0) {
	            var parts = line.split(" ");
	            jointStack[jointStack.length - 1]["offset"] = parts.slice(1);
	            for(x in jointStack[jointStack.length - 1]["offset"]){
	                jointStack[jointStack.length - 1]["offset"][x] = +jointStack[jointStack.length - 1]["offset"][x]
	            }
	        } else if (line.indexOf("CHANNELS") === 0) {
	            var parts = line.split(" ");
	            jointStack[jointStack.length - 1].setChannelNames(parts.slice(2));
	            jointStack[jointStack.length - 1]["channelLength"] = +parts[1];
	        } else if (line.indexOf("}") === 0) {
	            if (jointStack.length > 1) {
	                child = jointStack.pop();
	                jointStack[jointStack.length - 1].children.push(child);
	                child.parent = jointStack[jointStack.length - 1];
	
	                connectivityMatrix.push([child.parent, child])
	            }
	        } else if (line.indexOf("MOTION") == 0) {
	            return false;
	        }
	
	        return true;
	    };
	};
	
	BVHStreamParser.BVH = BVHStreamParser.BVH || {};
	
	BVHStreamParser.BVH.Joint = function (name, index) {
	
	    this.name = name;
	    this.children = [];
	    this.isEndSite = function () {
	        return this.children.length == 0;
	    };
	    this.rotationIndex = {};
	    this.positionIndex = {};
	
	    this.getChannels = function () {
	        var allChannels = [];
	        for (i = 0; i < this.skeleton.frameArray.length; i++) {
	            allChannels.push(this.getChannelsAt(i));
	        }
	        return allChannels;
	    };
	    this.getChannelsAt = function (frameNum) {
	        var channelsAtFrame = this.skeleton.frameArray[frameNum];
	        return channelsAtFrame.slice(this.channelOffset, this.channelOffset + this.channelLength);
	    };
	
	    this.setChannelNames = function (nameArr){
	        this.channelNames = nameArr;
	        for(i in this.channelNames){
	            var name = this.channelNames[i];
	            switch(name){
	                case "Xposition": this.positionIndex.x = i; break;
	                case "Yposition": this.positionIndex.y = i; break;
	                case "Zposition": this.positionIndex.z = i; break;
	
	                case "Xrotation": this.rotationIndex.x = i; break;
	                case "Yrotation": this.rotationIndex.y = i; break;
	                case "Zrotation": this.rotationIndex.z = i; break;
	            }
	        }
	    }
	};
	
	BVHStreamParser.BVH.Skeleton = function (root, map, arr, connectivityMatrix, frameCount, frameTime, frameArray) {
	    thisSkeleton = this;
	    this.root = root;
	    this.jointMap = map;
	    this.jointArray = arr;
	    this.connectivityMatrix = connectivityMatrix;
	    this.frameCount = frameCount;
	    this.frameTime = frameTime;
	    this.frameArray = frameArray;
	    this.bufferSize = 500;
	
	    for (i = 0; i < this.jointArray.length; i++) {
	        this.jointArray[i].skeleton = thisSkeleton;
	    }
	
	    this.fillFrameArray = function (fa) {
	        this.frameArray.push.apply(this.frameArray,fa);
	        //this.frameArray.push.apply(this.frameArray,fa);
	        
	        diff = this.frameArray.length - this.bufferSize;
	        // console.log('diff = ' + diff);
	        
	        /*
	        if (diff > 0) 
	            for (i=0;i<diff;i++)
	                this.frameArray.shift();
	
	        this.frameCount = this.frameArray.length;
	        */
	         
	        if (diff > 0) 
	            addedCount = this.frameCount;
	        else
	            addedCount = fa.length;
	
	        for(j=0; j < this.jointArray.length; j++){
	            var joint = this.jointArray[j];
	            updateWithPositionsSinceLast(joint, addedCount);
	        }
	        
	        return diff;
	    }
	
	    this.consumeFrames = function (index) {
	        for (i=0;i<=index;i++) {
	            this.frameArray.shift();
	            for (j=0;j<this.jointArray.length;j++)
	                this.jointArray[j].channels.shift();
	        }
	        this.frameCount = this.frameArray.length;
	    }
	
	    this.getChannels = function () {
	        return frameArray;
	    };
	    this.getChannelsAt = function (frameNum) {
	    	//How do I know which column is what?
	        //Why do you need the column index?
	        return frameArray[frameNum];
	    };
	    this.getFrameRate = function () {
	        return frameCount / frameTime;
	    };
	    this.getSkeleton = function () {
	        return root;
	    };
	
	    this.getHeadJoint = function () {
	    	// do a quick search in the joint names to see if any of them matches head, else return the something!!!!
	        return jointMap["Head"];
	    };
	    this.getPositionsAt = function (frameNum) {
	    	//for each joint, calculate its position in XYZ
	        //return an array of joints, each with .x, .y, and .z properties
	    	posFrame = [];
	
	    	for (j=0;j<this.jointArray.length;j++) {
	    		posFrame.push(this.jointArray[j].positions[frameNum]);
	    	}
	
	    	posFrame = posFrame.map(function(d) {
				return {
					x : d[0],
					y : d[1],
					z : d[2],
				};
			});
	
	        return posFrame;
	    };
	    this.getTPose = function () {
	    	// This function is basically the same as the getPositionsAt except that all the rotations will be 0
	        console.log("Not yet implemented");
	    };
	
	    function updatePositions(rootOffset, removeRoot, orientation, camera) {
	      //TODO: compelte the specification of this
	
	      for(j=0; j < this.jointArray.length; j++){
	          var joint = this.jointArray[j];
	          updateWithPositions(joint);
	      }
	    }
	
	    function updateWithPositions(joint){
	        var channelNames = joint.channelNames;
	        joint.channels = joint.getChannels();
	        joint.rotations = [];
	        joint.positions = [];
	        joint.rotmat = [];
	        for(i in joint.channels){
	            var channel = joint.channels[i];
	            var xpos = channel[joint.positionIndex.x] || 0,
	            ypos =  channel[joint.positionIndex.y] || 0,
	            zpos =  channel[joint.positionIndex.z] || 0;
	            // xangle =  deg2rad(channel[joint.rotationIndex.x] || 0),
	            // yangle =  deg2rad(channel[joint.rotationIndex.y] || 0),
	            // zangle= deg2rad(channel[joint.rotationIndex.z] || 0);
	
	            var posMatrix = [xpos, ypos, zpos];
	
	            if(!joint.parent){
	                //its the root
	                joint.positions[i] = posMatrix;//vectorAdd(joint.offset , posMatrix);
	                // ^ we can safely ignore the root's offset
	            }
	        }
	    }
	
	    function updateWithPositionsSinceLast(joint, addedCount){
	        var channelNames = joint.channelNames;
	        joint.channels = joint.getChannels();
	        joint.rotations = [];
	        joint.positions = [];
	        joint.rotmat = [];
	        for(i=joint.channels.length - addedCount;i < joint.channels.length; i++){
	            var channel = joint.channels[i];
	            var xpos = channel[joint.positionIndex.x] || 0,
	            ypos =  channel[joint.positionIndex.y] || 0,
	            zpos =  channel[joint.positionIndex.z] || 0;
	            // xangle =  deg2rad(channel[joint.rotationIndex.x] || 0),
	            // yangle =  deg2rad(channel[joint.rotationIndex.y] || 0),
	            // zangle= deg2rad(channel[joint.rotationIndex.z] || 0);
	
	            var posMatrix = [xpos, ypos, zpos];
	
	            if(!joint.parent){
	                //its the root
	                joint.positions[i] = posMatrix;//vectorAdd(joint.offset , posMatrix);
	                // ^ we can safely ignore the root's offset
	            }
	        }
	    }
	
	    function deg2rad(deg){
	        return deg * (Math.PI/180);
	    }
	};
	
	module.exports = BVHStreamParser;

/***/ },
/* 5 */
/***/ function(module, exports) {

	var C3DCharacter = C3DCharacter || {};
	
	C3DCharacter = function(n, jm, jg){
		this.name = n;
		
		this.markerMaterial = jm;
		this.makeMarkerGeometryFCN = jg;
	
		this.originPosition = new THREE.Vector3(0,0,0);
	
		this.markerdata = [];
		this.ready = false;
		this.scale = 0.5;
		this.markerMeshes = [];
	
		this.frameTime = 1/30;
		this.frameCount = 0;
	
		this.animIndex = 0;
		this.animStartTimeRef = 0;
		this.animOffset = 0;
		this.playing = true;
	
		this.debug = true;
	
		var self = this;
	
		//
	
		this.log = function(m) {
			if (self.debug)
				console.log(self.name + ": "+m.toString());
		};
	
		this.loadFromURL = function(url, callback) {
			self.log("Loading the mocap file ...");
			Pace.start();
			url2 = "../" + url;	
			self.url = url;

			Papa.parse(url2, {
				worker: true,
				delimiter: ",",	
				dynamicTyping: true,
				download: true,
				header: false,
				complete: function(results) {
					self.processData(results);
					if (callback)
						callback();
				}
			});
		};

		this.loadFromBuffer = function(data, callback) {
			self.log("Loading the mocap from buffer...");
			Pace.start();	
			var preData = data.split('\n');
			preData = preData.map(function(d,i){
				var cols = d.split(',');
				var floats = cols;
				// console.log(i);
				if (i!=0) {
					floats = cols.map(function(p, j){
						return parseFloat(p);
					});
				}
				
				return floats;
			});
			preData.pop();			
			
			this.processData({data: preData});
			if (callback)
				callback();
		}
	
		this.processData = function(results) {
			//self.markerdata = results.data;
			// console.log(results);

			for (i=0;i<results.data[0].length-3;i+=3) {
				var markerMesh = new THREE.Mesh(self.makeMarkerGeometryFCN(results.data[0][i], self.scale), self.markerMaterial);
				markerMesh.markerIndex = i;
				markerMesh.name = results.data[0][i];
				scene.add(markerMesh);
				self.markerMeshes.push(markerMesh);				
			}	
				
			self.markerNames = results.data[0];

			for (f=1;f<results.data.length;f++) {
				self.markerdata[f-1] = [];
				for (m=0;m<results.data[f].length-3;m+=3) {
					marker = {};
					marker.x = results.data[f][m];
					marker.y = results.data[f][m+1];
					marker.z = results.data[f][m+2];
					marker.name = self.markerNames[m];

					self.markerdata[f-1].push(marker);
				}								
			}						

			self.frameCount = self.markerdata.length;
			self.log("Done parsing!");	
			self.ready = true;
		}

		this.setOriginPosition = function (x, y, z) {
			self.originPosition.set(x,y,z);
		};
	
		this.setSkeletonScale = function(s) {
			self.rootMeshes.forEach(function (c) {
				c.scale.set(s,s,s);
			});
			self.jointMeshes[0].scale.set(s,s,s);
			self.jointMeshes[0].position.multiplyScalar(s);
		};
	
	
		this.animFrame = function (frame) {
			for (m=0;m<self.markerMeshes.length; m++) {
				self.markerMeshes[m].position.set(
					self.markerdata[frame][m].x * self.scale + self.originPosition.x,
					self.markerdata[frame][m].y * self.scale + self.originPosition.y,
					self.markerdata[frame][m].z * self.scale + self.originPosition.z);
			}
		};
	};
	
	module.exports = C3DCharacter;

/***/ }
/******/ ]);
//# sourceMappingURL=mocapjs.js.map
