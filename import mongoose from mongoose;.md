**import mongoose from "mongoose";**

**import bcrypt from "bcryptjs";**

**import jwt from "jsonwebtoken";**



**const userSchema = new mongoose.Schema({**

**fullname:     { type: String, required: true },**

**username:     { type: String, required: true, unique: true, index: true },**

**email:        { type: String, required: true, unique: true, index: true },**

**password:     { type: String, required: true },**

**profileimage:   { type: String , default : "url"},**

**roles:        { type: \[String], enum: \['user', 'admin'], default: \['user'] },**

**subscription: { type: String, enum: \['free', 'premium'], default: 'free' },**

**isActive:     { type: Boolean, default: true },**

**blockedUsers:      \[{ type: mongoose.Schema.Types.ObjectId, ref: "User" }],**

**followers:         \[{ type: mongoose.Schema.Types.ObjectId, ref: "User" }],**

**following:         \[{ type: mongoose.Schema.Types.ObjectId, ref: "User" }]**

**}, { timestamps: true });**







**export const User = mongoose.model("User", userSchema);**







const videoSchema = new mongoose.Schema({

  title:       { type: String, required: true },

  description: { type: String },

  videoUrl:   { type: String, required: true },

  thumbnail:   { type: String, required: true },

  owner:       { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, index: true },

  views:       { type: Number, default: 0 },

  isPublic: { type: Boolean, default: true },

  isPremium:   { type: Boolean, default: false },

  category:    { type: String, enum: \["Education", "Music", "Technology", "Entertainment", "Lifestyle", "General"], default: "General" }



}, { timestamps: true });



export const Video = mongoose.model("Video", videoSchema);









import mongoose from "mongoose";



const likeSchema = new mongoose.Schema({



 video : { type :

  likeBy:   { type: mongoose.Schema.Types.ObjectId, ref : User ,required: true, index: true },

 likesCount:      { type: Number, default: 0 },



 

}); timestamp : true



likeSchema.index({ userId: 1, targetType: 1, targetId: 1 }, { unique: true });



export const Like = mongoose.model("Like", likeSchema);









**const commentSchema = new Schema({**

**content:{**

&nbsp;   \*\*type:String,\*\*

    \*\*required:true\*\*

    \*\*},\*\*

    \*\*video:{\*\*

        \*\*type:Schema.Types.ObjectId,\*\*

        \*\*ref:"Video",\*\*

    \*\*},\*\*

    \*\*owner:{\*\*

        \*\*type:Schema.Types.ObjectId,\*\*

        \*\*ref:"User",\*\*

    \*\*}\*\*


**},{timestamps:true})**





**subscriber:{**

&nbsp;   \*\*type: Schema.Types.ObjectId,\*\*

    \*\*ref:"Users"\*\*


**},**

**channel:{**

&nbsp;   \*\*type: Schema.Types.ObjectId,\*\*

    \*\*ref:"Users"\*\*


**},**

**},{timestamps  :true})**





**import mongoose from "mongoose";**



**const playlistSchema = new mongoose.Schema({**

**userId:   { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, index: true },**

**playlistname:     { type: String, required: true },**

**isPublic: { type: Boolean, default: false },**

**videos:   \[{ type: mongoose.Schema.Types.ObjectId, ref: "Video" }]**

**}, { timestamps: true });**



**export const Playlist = mongoose.model("Playlist", playlistSchema);**





**import mongoose from "mongoose";**



**const notificationSchema = new mongoose.Schema({**

**userId:      { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, index: true },**

**title:       { type: String },**

**description: { type: String },**

**targetId:    { type: mongoose.Schema.Types.ObjectId },**

**type:        { type: String, enum: \["report", "comment", "like", "moderation", "general"] },**

**isRead:      { type: Boolean, default: false }**

**}, { timestamps: true });**



**notificationSchema.index({ userId: 1, isRead: 1 });**



**export const Notification = mongoose.model("Notification", notificationSchema);**



**import mongoose from "mongoose";**



**const reportSchema = new mongoose.Schema({**

**reporterId:  { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },**

**targetType:  { type: String, enum: \["video", "comment", "user"], required: true },**

**targetId:    { type: mongoose.Schema.Types.ObjectId, required: true },**

**reason:      { type: String, required: true },**

**status:      { type: String, enum: \["pending", "reviewed", "resolved", "rejected"], default: "pending", index: true },**

**adminAction: {**

&nbsp;   \*\*adminId: { type: mongoose.Schema.Types.ObjectId, ref: "Admin" },\*\*

    \*\*action:  String,\*\*

    \*\*takenAt: Date\*\*


**}**

**}, { timestamps: true });**



**reportSchema.index({ status: 1, createdAt: -1 });**



**export const Report = mongoose.model("Report", reportSchema);**





**import mongoose from "mongoose";**



**const adminSchema = new mongoose.Schema({**

**userId:     { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, unique: true, index: true },**

**role:       { type: String, required: true },**

**permissions:\[{ type: String }],**

**lastLogin:  { type: Date }**

**}, { timestamps: true });**



**export const Admin = mongoose.model("Admin", adminSchema);**





**import mongoose from "mongoose";**



**const settingSchema = new mongoose.Schema({**

**name:        { type: String, unique: true, required: true },**

**description: { type: String },**

**profilecustomeimage : {type : string**

**updatedBy:   { type: mongoose.Schema.Types.ObjectId, ref: "user" },**

**updatedAt:   { type: Date, default: Date.now }**

**});**



**export const Setting = mongoose.model("Setting", settingSchema);**







**Deletion Log Schema (deletionLog.model.js)**



**import mongoose from "mongoose";**



**const deletionLogSchema = new mongoose.Schema({**

**contentType: { type: String, enum: \["video", "comment"], required: true },**

**originalId:  { type: mongoose.Schema.Types.ObjectId, required: true },**

**deletedBy:   { type: mongoose.Schema.Types.ObjectId, ref: "Admin" },**

**reason:      { type: String },**

**snapshot:    mongoose.Schema.Types.Mixed**

**}, { timestamps: true });**



**export const DeletionLog = mongoose.model("DeletionLog", deletionLogSchema);**







**history log schema (**

**watchHistory:      \[{ type: mongoose.Schema.Types.ObjectId, ref: "Video" }],   /// playlist**

**watchLater:        \[{ type: mongoose.Schema.Types.ObjectId, ref: "Video" }],  ///**

