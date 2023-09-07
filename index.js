"use strict";
const image = document.getElementById('image');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const blurCanvas = document.createElement('canvas');
const blurCtx = blurCanvas.getContext('2d');
const pixelCanvas = document.createElement('canvas');
const pixelCtx = pixelCanvas.getContext('2d');
const brightCanvas = document.createElement('canvas');
const brightCtx = brightCanvas.getContext('2d');
const gameCanvas = document.getElementById('game');
const gameCtx = gameCanvas.getContext('2d');
const markRegions = document.getElementById('markregions');
const fileSelect = document.getElementById("fileSelect");
const fileElem = document.getElementById("fileElem");
const retry = document.getElementById("retry");
const type = document.getElementById("type");
const setting = document.getElementById("setting");
const taskContainer = document.getElementById("task-container");
const task = document.getElementById("task");
const done = document.getElementById("done");
const clickedInput = document.getElementById('clickedBox')

let tasksCpy

const t = {};
let model;

const log = (msg) => console.log(msg);

const options = {
    modelPath: './models/default-f16/model.json',
    imagePath: './samples/example.jpg',
    minScore: 0.38,
    maxResults: 50,
    iouThreshold: 0.5,
    outputNodes: ['output1', 'output2', 'output3'],
    resolution: [801, 1112],
    user: {
        map: {
            blur,
            solid: solidColor,
            bright,
            pixel
        },
        blurStrength: 10,
        brightness: 3.5,
        pixelSize: 3.5,
        fillColor: '#000000',
        censorType: 'pixel'
    }
};

const labels = [
    'exposed anus',  //0
    'exposed armpits',  //1
    'belly',  //2
    'exposed belly',  //3
    'buttocks',  //4
    'exposed buttocks',  //5
    'female face',  //6
    'male face',  //7
    'feet',  //8
    'exposed feet',  //9
    'breast',  //10
    'exposed breast',  //11
    'vagina',  //12
    'exposed vagina',  //13
    'male breast',  //14
    'exposed male breast',  //15
];

const betaSettings = {
    pathetic: {
        person: [1,2,3,4,6,7,8,9,14,15],
        sexy: [5,10,12],
        nude: [0,11,13],
    },
    original: {
        person: [1,2,3,4,6,7,8,9,14,15],
        sexy: [1,3,5,9,10,12,14,15],
        nude: [0,11,13],
    },
    ultimate: {
        person: [7,14,15],
        sexy: [1,2,4,6,8,10],
        nude: [0,3,5,9,11,12,13],
    }
}

let composite = betaSettings.pathetic

async function processPrediction(boxesTensor, scoresTensor, classesTensor, inputTensor) {
    const boxes = await boxesTensor.array();
    const scores = await scoresTensor.data();
    const classes = await classesTensor.data();
    const nmsT = await tf.image.nonMaxSuppressionAsync(boxes[0], scores, options.maxResults, options.iouThreshold, options.minScore); // sort & filter results
    const nms = await nmsT.data();
    tf.dispose(nmsT);
    const parts = [];
    for (const i in nms) { // create body parts object
        const id = parseInt(i);
        parts.push({
            score: scores[i],
            id: classes[id],
            class: labels[classes[id]],
            box: [
                Math.trunc(boxes[0][id][0]),
                Math.trunc(boxes[0][id][1]),
                Math.trunc((boxes[0][id][3] - boxes[0][id][1])),
                Math.trunc((boxes[0][id][2] - boxes[0][id][0])),
            ],
        });
    }
    const result = {
        input: { width: inputTensor.shape[2], height: inputTensor.shape[1] },
        person: parts.filter((a) => composite.person.includes(a.id)).length > 0,
        sexy: parts.filter((a) => composite.sexy.includes(a.id)).length > 0,
        nude: parts.filter((a) => composite.nude.includes(a.id)).length > 0,
        parts,
    };
    return result;
}

function blur({left=0, top=0, width=0, height=0}) {
    if (width === 0 || height === 0 || canvas.width === 0 || canvas.height === 0)
        return;
    blurCanvas.width = width
    blurCanvas.height = height

    blurCtx.filter = `blur(${options.user.blurStrength}px)`;

    blurCtx.drawImage(canvas, left, top, width, height, 0, 0, width ,height);

    ctx.drawImage(blurCanvas, left, top, width, height);
}

function bright({left=0, top=0, width=0, height=0}) {
    if (width === 0 || height === 0 || canvas.width === 0 || canvas.height === 0)
        return;
    brightCanvas.width = width
    brightCanvas.height = height

    brightCtx.filter = `brightness(${options.user.brightness})`;

    brightCtx.drawImage(canvas, left, top, width, height, 0, 0, width ,height);

    ctx.drawImage(brightCanvas, left, top, width, height);
}

function pixel({left=0, top=0, width=0, height=0}) {
    if (width === 0 || height === 0 || canvas.width === 0 || canvas.height === 0)
        return;
    pixelCanvas.width = width
    pixelCanvas.height = height

    pixelCtx.drawImage(canvas, left, top, width, height, 0, 0, width ,height);

    let size = options.user.pixelSize / 100,
    w = pixelCanvas.width * size,
    h = pixelCanvas.height * size;

    pixelCtx.drawImage(pixelCanvas, 0, 0, w, h);

    pixelCtx.msImageSmoothingEnabled = false;
    pixelCtx.mozImageSmoothingEnabled = false;
    pixelCtx.webkitImageSmoothingEnabled = false;
    pixelCtx.imageSmoothingEnabled = false;

    pixelCtx.drawImage(pixelCanvas, 0, 0, w, h, 0, 0, pixelCanvas.width, pixelCanvas.height);

    ctx.drawImage(pixelCanvas, left, top, width, height);

    pixelCtx.msImageSmoothingEnabled = true;
    pixelCtx.mozImageSmoothingEnabled = true;
    pixelCtx.webkitImageSmoothingEnabled = true;
    pixelCtx.imageSmoothingEnabled = true;
}

function solidColor({left=0, top=0, width=0, height=0}) {
    if (width === 0 || height === 0 || canvas.width === 0 || canvas.height === 0)
        return;
    ctx.fillStyle = options.user.fillColor
    ctx.fillRect(left, top, width, height)
}

function rect({ x=0, y=0, width=0, height=0, radius=8, lineWidth=2, color='white', title='', font='20px "Segoe UI"'}) {
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.lineWidth = 4;
    ctx.fillStyle = color;
    ctx.font = font;
    ctx.fillText(title, x + 4, y - 4);
}

function processParts(res) {
    for (const obj of res.parts) { // draw all detected objects
        if (composite.nude.includes(obj.id))
            options.user.map[options.user.censorType]({ left: obj.box[0], top: obj.box[1], width: obj.box[2], height: obj.box[3] });
        if (composite.sexy.includes(obj.id))
            options.user.map[options.user.censorType]({ left: obj.box[0], top: obj.box[1], width: obj.box[2], height: obj.box[3] });
        if (markRegions.checked) {
            rect({ x: obj.box[0], y: obj.box[1], width: obj.box[2], height: obj.box[3], title: `${obj.class}` });
        }
    }
}

async function processLoop() {
    if (canvas.width !== image.width)
        canvas.width = image.width;
    if (canvas.height !== image.height)
        canvas.height = image.height;
    if (canvas.width > 0 && model) {

        t.buffer = await tf.browser.fromPixelsAsync(image);
        t.resize = (options.resolution[0] > 0 && options.resolution[1] > 0 && (options.resolution[0] !== image.width || options.resolution[1] !== image.height)) // do we need to resize
            ? tf.image.resizeNearestNeighbor(t.buffer, [options.resolution[1], options.resolution[0]])
            : t.buffer;
        t.cast = tf.cast(t.resize, 'float32');
        t.batch = tf.expandDims(t.cast, 0);

        [t.boxes, t.scores, t.classes] = await model.executeAsync(t.batch, options.outputNodes);

        const res = await processPrediction(t.boxes, t.scores, t.classes, t.cast);
        await tf.browser.toPixels(t.resize, canvas);
        processParts(res);
    }
}

async function main() {
    if (tf.engine().registryFactory.webgpu && navigator?.gpu)
        await tf.setBackend('webgpu');
    else
        await tf.setBackend('webgl');
    tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true); // doubles the performance
    await tf.ready();

    model = await tf.loadGraphModel(options.modelPath);
    image.src = options.imagePath;
    image.onload = async () => {
        options.resolution[0] = image.width
        options.resolution[1] = image.height

        await processLoop()
        gameSetup()
    };
}

fileSelect.addEventListener("click", (e) => {
    if (fileElem) {
        fileElem.click();
    }
},false);

function reset() {
    tasks.taskArr = [...tasksCpy]
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    pixelCtx.clearRect(0, 0, pixelCanvas.width, pixelCanvas.height);
    blurCtx.clearRect(0, 0, blurCanvas.width, blurCanvas.height);
    brightCtx.clearRect(0, 0, brightCanvas.width, brightCanvas.height);
    gameCtx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);
    taskContainer.style.display = 'none'
    done.disabled = false
    gameCanvas.addEventListener('click', clickFunc, false)
    boxes = {}
    main()
}

retry.addEventListener('click', () => {
    reset()
})

fileElem.onchange = function(){
    options.imagePath = window.URL.createObjectURL(this.files[0])
    reset()
}

type.onchange = (e) => {
    options.user.censorType = e.target.value
}

setting.onchange = (e) => {
    composite = betaSettings[e.target.value]
}

const tasks = {
    endArr: [
        'Denied. Loser!',
        'Ruin your orgasm like a good beta',
        'Lucky you, full orgasm',
        'Denied until someone gives you permission to cum',
    ],
    taskArr: [
        'Edge X times', //0
        'Slap balls X times', //1
        'Use 2 fingers to stroke', //2
        'Taste your precum', //3
        'Moan like a good little slut', //4
        'Watch 5 minutes of BBC porn without touching', //5
        'Make an ahegao face while you jerk off for 2 minutes', //6
        'Use your non dominant hand to stroke', //7
        'Put your used underwear in your mouth for 5 minutes', //8
        'Tie your pathetic balls up', //9
        'Stroke at 60 BPM for the rest of the game', //10
        'Find a beta safe wallpaper and have it set for 1 week', //11
        'Spank your ass X times hard', //12
        'stroke just the tip of your pathetic clit', //13
        'Slap your face with your precum', //14
        'Play a game of chess, slap your balls for every piece taken off the board', //15
        'Put clips on your nipples', //16
        'Write "Bad Slut" on your body', //17
        'Insert a butt plug', //18
        'You can undo one task if applicable', //19
        'Write "Free use slut" on your ass cheek', //20
        'On your knees, bitch', //21
        'Spit on your hand and use it as lube', //22
    ],
    useOnce: [10,11,15,16,17,18,19,20,21],
    orgasmType: ''
}

class Box {
    constructor(task=null, left, top, w, h, id) {
        this.task = task;
        this.left = left;
        this.top = top;
        this.w = w;
        this.h = h;
        this.color = '#ff20da';
        this.id = id
        this.text=false
    }

    drawBox() {
        gameCtx.rect(this.left, this.top, this.w, this.h);
        gameCtx.fillStyle = this.color;
        gameCtx.fill()
        gameCtx.lineWidth = 1;
        gameCtx.strokeStyle = 'black';
        gameCtx.stroke();

        if (this.text) {
            this.drawText()
        }
    }

    pauseGame(gameOver=null) {
        gameCanvas.removeEventListener('click', clickFunc)
        taskContainer.style.display = 'flex'
        task.innerText += ' ' + this.task

        if (gameOver) {
            done.disabled = true
            clickedInput.value = this.id + ' over'
        } else {
            clickedInput.value = this.id
        }
    }

    drawText() {
        if (!this.task) {
            removeTile(this.id)
            delete boxes[this.id]
            if (tasks.endArr.includes(this.task)) {
                this.pauseGame('over')
            }
            return
        };
        if (this.text) return;

        if (tasks.endArr.includes(this.task)) {
            this.pauseGame('over')
        } else {
            this.pauseGame()
        }

        this.text = true
    }
}

let boxes = {}

function removeTile(boxId) {
    let boxDelete = boxes[boxId]

    let newCanvas = document.createElement('canvas')
    let newCtx = newCanvas.getContext('2d')

    newCanvas.width = boxDelete.w
    newCanvas.height = boxDelete.h

    newCtx.drawImage(canvas, boxDelete.left, boxDelete.top, boxDelete.w, boxDelete.h, 0, 0, boxDelete.w, boxDelete.h)
    gameCtx.drawImage(newCanvas, boxDelete.left, boxDelete.top, boxDelete.w, boxDelete.h)

    delete boxes[boxId]
}

done.addEventListener('click', () => {
    let parts = clickedInput.value.split(' ')

    clickedInput.value = parts[0]

    if (!parts[1]) {
        gameCanvas.addEventListener('click', clickFunc, false)
        taskContainer.style.display = 'none'
        task.innerText = 'Current Task: '
    } else {
        taskContainer.style.opacity = 0.3
    }

    removeTile(clickedInput.value)

    clickedInput.value = ''
})

function gameSetup() {
    tasksCpy = [...tasks.taskArr]

    gameCanvas.width = image.width
    gameCanvas.height = image.height

    let smallestSide = gameCanvas.width <= gameCanvas.height ? gameCanvas.width : gameCanvas.height
    let largestSide
    let direction
    if (gameCanvas.width > gameCanvas.height) {
        largestSide = gameCanvas.width
        direction = 'landscape'
    } else {
        largestSide = gameCanvas.height
        direction = 'portrait'
    }

    let ratio = largestSide/smallestSide
    let small
    let large
    if (direction == 'portrait') {
        small = (14.5 / 100) * smallestSide // 14.5% of smallest side
        large = small*ratio
    } else {
        small = (14.5 / 100) * largestSide
        large = small/ratio
    }

    let left = 0;
    let rows, cols
    rows = gameCanvas.height / large
    cols = gameCanvas.width / small

    let top = 0;

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let id = i.toString()+j.toString()
            let box = new Box('', left, top, small, large, id)
            boxes[id] = box
            left += small
        }
        left = 0
        top += large
    }

    for (const [key,value] of Object.entries(boxes)) {
        let chance = Math.random()

        if (chance < 0.50) {
            let randNum = Math.floor(Math.random()*tasks.taskArr.length)
            let boxTask = tasks.taskArr[randNum]

            if (tasks.useOnce.includes(randNum)) {
                tasks.taskArr.splice(randNum, 1)
                console.log(boxTask)
            }

            boxes[key].task = boxTask.replace('X', Math.ceil(Math.random()*15))
        }

        value.drawBox()
    }

    let chances = Math.floor(Math.random()*tasks.endArr.length)+1
    let count = 0
    for (let i = 0; i < chances; i++) {
        while (count<chances) {
            let keys = Object.keys(boxes)
            let randKey = keys[ keys.length * Math.random() << 0];

            if (boxes[randKey].task == '') {
                boxes[randKey].task = tasks.endArr[Math.floor(Math.random()*tasks.endArr.length)]
                count++
            }
        }
    }
}

function collides(x,y) {
    for (const [key,value] of Object.entries(boxes)) {
        let left = value.left
        let right = value.left+value.w;
        let top = value.top
        let bottom = value.top+value.h;
        if (right >= x
            && left <= x
            && bottom >= y
            && top <= y) {
            return key;
        }
    }
}

function clickFunc(e) {
    let boxId = collides(e.offsetX, e.offsetY);

    boxes[boxId].drawText()
}

gameCanvas.addEventListener('click', clickFunc, false)

main()